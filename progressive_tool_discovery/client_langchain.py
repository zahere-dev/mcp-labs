"""
MCP client demonstrating the "progressive tool discovery" pattern described in
https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices

This variant is built on LangChain's agent framework (`langchain.agents.create_agent`)
instead of a hand-rolled OpenAI chat-completions loop -- see
client_with_keyword_search.py / client_with_embeddings_search.py for the raw
OpenAI SDK versions. The Layer 1 search strategy is the same plain keyword
matching as client_with_keyword_search.py; only the agent framework differs.

Two LangChain-specific pieces are worth calling out:

  - In `direct` mode, the real MCP tool schemas are converted into LangChain
    tools with a single call to `langchain_mcp_adapters.tools.load_mcp_tools`
    -- no manual JSON-schema-to-tool conversion needed.
  - In `progressive` mode, the three meta-tools (search_tools,
    get_tool_details, call_tool) are plain Python async functions decorated
    with `@tool`, closing over the host-side `ToolCatalog` and MCP
    `ClientSession`. `create_agent` wires up the tool-calling loop (including
    executing tool calls and feeding results back to the model) for us --
    there's no manual message-append loop like in the raw OpenAI clients.
  - Every AIMessage LangChain returns carries `usage_metadata` (actual
    input/output/total tokens reported by the model). `run()` sums
    `total_tokens` across every LLM call in the run (one call per turn, in
    either mode) and prints it -- the real token cost actually spent, as
    opposed to the *estimate* used only for the direct-vs-progressive mode
    decision. Compare this number between `--mode direct` and `--mode
    progressive` to see the token savings in practice.

Run the server first:
    python server.py

Then:
    python client_langchain.py
    python client_langchain.py --mode direct "what's the weather in Paris?"
    python client_langchain.py --mode progressive "close ISSUE-1 with resolution 'fixed'"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.tools import load_mcp_tools
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as MCPTool

load_dotenv()

SERVER_URL = "http://127.0.0.1:8300/mcp"
MODEL = "gpt-4o-mini"
CONTEXT_WINDOW_TOKENS = 200_000
# Percentage of the context window that the *full* tool catalog is allowed to
# consume before we switch from "dump everything" to progressive discovery.
PROGRESSIVE_DISCOVERY_THRESHOLD_PCT = 1.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) -- good enough for a threshold check."""
    return max(1, len(text) // 4)


def full_schema_blob(tool: MCPTool) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description or "", "input_schema": tool.inputSchema}


@dataclass
class ToolCatalog:
    """Host-side cache of every tool a connected server exposes.

    This is populated from tools/list and re-populated on list_changed
    notifications. It is never sent to the model wholesale -- the model only
    ever sees what search_tools / get_tool_details choose to reveal.

    Per the 2026-07-28 spec, tools/list responses can carry `ttlMs` /
    `cacheScope` hints in `_meta` (see server.py). is_stale()/mark_stale()
    track those hints; ProgressiveDiscoveryAgent.ensure_fresh() does the
    actual re-fetch.
    """

    tools: dict[str, MCPTool] = field(default_factory=dict)
    ttl_ms: int | None = None
    cache_scope: str | None = None
    fetched_at: float = 0.0
    stale: bool = True

    def refresh(self, tools: list[MCPTool], meta: dict[str, Any] | None = None) -> None:
        self.tools = {t.name: t for t in tools}
        meta = meta or {}
        self.ttl_ms = meta.get("ttlMs")
        self.cache_scope = meta.get("cacheScope")
        self.fetched_at = time.monotonic()
        self.stale = False
        logger.info(
            "Tool catalog refreshed: {} tools cached host-side (ttlMs={}, cacheScope={})",
            len(self.tools),
            self.ttl_ms,
            self.cache_scope,
        )

    def mark_stale(self) -> None:
        """Invalidate the cache without re-fetching -- the agent's
        ensure_fresh() does the actual tools/list round trip lazily."""
        self.stale = True

    def is_stale(self) -> bool:
        if self.stale:
            return True
        if self.ttl_ms is None:
            return False
        return (time.monotonic() - self.fetched_at) * 1000 > self.ttl_ms

    def full_definitions_token_estimate(self) -> int:
        blob = json.dumps([full_schema_blob(t) for t in self.tools.values()])
        return estimate_tokens(blob)

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Layer 1: keyword search over tool name + description.

        Same strategy as client_with_keyword_search.py -- see that file for
        the embeddings-based alternative in client_with_embeddings_search.py.
        """
        stopwords = {
            "a", "an", "the", "to", "of", "for", "and", "or", "with", "is",
            "on", "in", "at", "by", "it", "its", "if", "so", "do", "be",
        }
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2 and t not in stopwords]
        scored: list[tuple[int, MCPTool]] = []
        for t in self.tools.values():
            haystack = f"{t.name} {t.description or ''}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score, t))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [t for _, t in scored[:limit]]
        if not results:
            # graceful fallback: surface everything with a name-only view
            # rather than leaving the model with nothing to work with.
            results = list(self.tools.values())[:limit]
        return [{"name": t.name, "description": t.description or ""} for t in results]

    def get_details(self, name: str) -> dict[str, Any]:
        """Layer 2: return the full schema for exactly one tool."""
        t = self.tools.get(name)
        if t is None:
            return {"error": f"no such tool: {name}"}
        return full_schema_blob(t)


class ProgressiveDiscoveryAgent:
    """Drives a LangChain `create_agent` graph against an MCP server's tools,
    choosing between naive (load-everything) and progressive discovery based
    on how much context the full tool catalog would consume.
    """

    def __init__(self, session: ClientSession, mode: str = "auto"):
        self.session = session
        self.mode = mode
        self.catalog = ToolCatalog()
        self.model: str | BaseChatModel = MODEL
        self.agent = None

    async def bootstrap(self) -> str:
        result = await self.session.list_tools()
        self.catalog.refresh(result.tools, result.meta)
        active_mode = self._decide_mode()
        tools = await self._build_tools()
        self.agent = create_agent(
            self.model,
            tools,
            system_prompt=(
                "You are a helpful assistant with access to tools. "
                "If a tool you need isn't directly available, use search_tools "
                "to find it, get_tool_details to inspect its schema, then call_tool "
                "to execute it."
            ),
        )
        return active_mode

    async def on_server_notification(self, notification) -> None:
        # Invalidate the cache immediately -- per the caching guidelines, a
        # list_changed notification makes the cache stale even if its TTL
        # hasn't lapsed yet.
        from mcp.types import ToolListChangedNotification

        if isinstance(notification.root, ToolListChangedNotification):
            logger.info("Received tools/list_changed -- marking catalog stale")
            self.catalog.mark_stale()
            if self.active_mode == "direct":
                # Direct mode has no search/inspect layer to lazily refresh
                # through -- the LangChain tool list itself needs updating,
                # so re-fetch right away instead of waiting for next use.
                await self.ensure_fresh()

    async def ensure_fresh(self) -> None:
        if not self.catalog.is_stale():
            return
        logger.info("Tool catalog cache expired or invalidated -- re-fetching tools/list")
        result = await self.session.list_tools()
        self.catalog.refresh(result.tools, result.meta)
        if self.active_mode == "direct":
            tools = await self._build_tools()
            self.agent = create_agent(self.model, tools)

    def _decide_mode(self) -> str:
        if self.mode in ("direct", "progressive"):
            active = self.mode
        else:
            tokens = self.catalog.full_definitions_token_estimate()
            threshold_tokens = CONTEXT_WINDOW_TOKENS * PROGRESSIVE_DISCOVERY_THRESHOLD_PCT / 100
            active = "progressive" if tokens > threshold_tokens else "direct"
            logger.info(
                "Full tool definitions ~{} tokens ({:.2f}% of {}-token window, threshold {:.1f}%) -> mode={}",
                tokens,
                100 * tokens / CONTEXT_WINDOW_TOKENS,
                CONTEXT_WINDOW_TOKENS,
                PROGRESSIVE_DISCOVERY_THRESHOLD_PCT,
                active,
            )
        self.active_mode = active
        return active

    async def _build_tools(self) -> list[BaseTool]:
        if self.active_mode == "direct":
            # langchain-mcp-adapters converts every real MCP tool schema into
            # a LangChain tool in one call -- this is the naive "load
            # everything" baseline.
            return await load_mcp_tools(self.session)
        return self._build_meta_tools()

    def _build_meta_tools(self) -> list[BaseTool]:
        # Closures over `self` so the meta-tools always see the current
        # catalog and session, even after a list_changed rebuild.
        agent = self
        catalog = self.catalog
        session = self.session

        @tool
        async def search_tools(query: str) -> list[dict[str, str]]:
            """Search the full catalog of available tools by keyword. Returns
            matching tool names with a one-line description each. Call this
            before calling any tool you haven't inspected yet."""
            await agent.ensure_fresh()
            return catalog.search(query)

        @tool
        async def get_tool_details(name: str) -> dict[str, Any]:
            """Fetch the full input schema and documentation for one specific
            tool by name, so you know exactly what arguments to pass."""
            await agent.ensure_fresh()
            return catalog.get_details(name)

        @tool
        async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
            """Execute a tool by name with the given arguments and return its result."""
            logger.info("-> tools/call {} {}", name, arguments)
            result = await session.call_tool(name, arguments=arguments)
            if result.isError:
                text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                return {"error": text}
            return "\n".join(c.text for c in result.content if hasattr(c, "text"))

        return [search_tools, get_tool_details, call_tool]

    async def run(self, user_message: str) -> tuple[str, int]:
        result = await self.agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
        messages = result["messages"]
        final_message = messages[-1]
        # Sum actual token usage the model reported across every LLM call in
        # this run (one per turn) -- not the same as the *estimate* used for
        # the direct-vs-progressive mode decision, this is the real cost.
        total_tokens = sum(
            m.usage_metadata["total_tokens"]
            for m in messages
            if isinstance(m, AIMessage) and m.usage_metadata
        )
        return final_message.content, total_tokens


async def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive tool discovery MCP client (LangChain)")
    default_query = "search for the contact jane, and if she doesn't exist create one"
    parser.add_argument(
        "query",
        nargs="?",
        default=default_query,
        help=f"Natural-language task for the agent (default: {default_query!r})",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "direct", "progressive"],
        default="auto",
        help="auto (default): pick based on estimated token cost. direct: always load full MCP tool "
        "schemas via langchain-mcp-adapters. progressive: always use search_tools/get_tool_details/call_tool.",
    )
    args = parser.parse_args()

    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream, _get_session_id):
        agent_ref: dict[str, ProgressiveDiscoveryAgent] = {}

        async def message_handler(message):
            from mcp.shared.session import RequestResponder

            if isinstance(message, RequestResponder):
                return
            if hasattr(message, "root"):
                agent = agent_ref.get("agent")
                if agent is not None:
                    await agent.on_server_notification(message)

        async with ClientSession(read_stream, write_stream, message_handler=message_handler) as session:
            info = await session.initialize()
            logger.info("Connected to {} v{}", info.serverInfo.name, info.serverInfo.version)

            agent = ProgressiveDiscoveryAgent(session, mode=args.mode)
            agent_ref["agent"] = agent
            active_mode = await agent.bootstrap()
            print(f"[client] discovery mode: {active_mode}")

            answer, total_tokens = await agent.run(args.query)
            print("\n=== Final answer ===")
            print(answer)
            print(f"\nTotal tokens used this run: {total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
