"""
MCP client demonstrating the "progressive tool discovery" pattern described in
https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices

This variant uses plain KEYWORD matching as the Layer 1 search strategy (see
client_with_embeddings_search.py for the embeddings-based variant).

Naive host behaviour is: fetch tools/list from every connected server and hand
every full tool schema to the model at the start of the conversation. That's
fine for a handful of tools, but wasteful once schemas eat a meaningful slice
of the context window.

This client instead:

  1. Connects to the server and calls tools/list as normal, but caches the
     full definitions host-side instead of injecting them into the model's
     context (`ToolCatalog`).
  2. Estimates how many tokens the full tool set would cost. If that is below
     a configurable percentage of the model's context window, it falls back
     to the naive "load everything" behaviour (small tool sets don't need
     progressive discovery). Otherwise it switches to progressive discovery.
  3. In progressive discovery mode, only three lightweight meta-tools are
     exposed to the model:
       - search_tools(query)      -> Layer 1: catalog / keyword search
       - get_tool_details(name)   -> Layer 2: inspect full schema on demand
       - call_tool(name, args)    -> Layer 3: execute
     The model never sees the real tool schemas in its `tools` array; it
     only sees them as JSON text returned by get_tool_details. Because the
     `tools` array handed to the API never changes shape, prompt caching of
     the tool-definitions prefix stays intact across turns.
  4. The catalog honors the server's `ttlMs` / `cacheScope` cache hints
     (2026-07-28 spec, carried in tools/list's `_meta`): it re-fetches once
     the TTL lapses, and is marked stale immediately on a
     notifications/tools/list_changed notification even if the TTL hasn't
     expired yet -- the actual re-fetch happens lazily on next use.

Run the server first:
    python server.py

Then:
    python client_with_keyword_search.py "search for the contact jane, and if she doesn't exist create one"
    python client_with_keyword_search.py --mode direct "what's the weather in Paris?"
    python client_with_keyword_search.py --mode progressive "close ISSUE-1 with resolution 'fixed'"
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
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as MCPTool
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

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


def mcp_tool_to_openai_param(tool: MCPTool) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


@dataclass
class ToolCatalog:
    """Host-side cache of every tool a connected server exposes.

    This is populated from tools/list and re-populated on list_changed
    notifications. It is never sent to the model wholesale -- the model only
    ever sees what search_tools / get_tool_details choose to reveal.

    Per the 2026-07-28 spec, tools/list responses can carry `ttlMs` /
    `cacheScope` hints in `_meta` (see server.py). This cache honors them:
    it re-fetches once the TTL lapses, and treats itself as stale
    immediately on a list_changed notification even if the TTL hasn't
    expired yet (see the "Caching" section of the client-best-practices
    article).
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
        """Invalidate the cache without re-fetching -- the next call to
        ensure_fresh() will do the actual tools/list round trip."""
        self.stale = True

    def is_stale(self) -> bool:
        if self.stale:
            return True
        if self.ttl_ms is None:
            return False
        return (time.monotonic() - self.fetched_at) * 1000 > self.ttl_ms

    async def ensure_fresh(self, session: ClientSession) -> None:
        if not self.is_stale():
            return
        logger.info("Tool catalog cache expired or invalidated -- re-fetching tools/list")
        result = await session.list_tools()
        self.refresh(result.tools, result.meta)

    def full_definitions_token_estimate(self) -> int:
        blob = json.dumps([mcp_tool_to_openai_param(t) for t in self.tools.values()])
        return estimate_tokens(blob)

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Layer 1: keyword search over tool name + description.

        A real implementation might swap this for BM25, embeddings, or a
        subagent -- see the "Choosing a Discovery Strategy" section of the
        article. Keyword matching is simple, cheap, and works well when tool
        names/descriptions are descriptive, which is the common case.
        """
        stopwords = {
            "a", "an", "the", "to", "of", "for", "and", "or", "with", "is",
            "on", "in", "at", "by", "it", "its", "if", "so", "do", "be",
        }
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2 and t not in stopwords]
        scored: list[tuple[int, MCPTool]] = []
        for tool in self.tools.values():
            haystack = f"{tool.name} {tool.description or ''}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score, tool))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [t for _, t in scored[:limit]]
        if not results:
            # graceful fallback: surface everything with a name-only view
            # rather than leaving the model with nothing to work with.
            results = list(self.tools.values())[:limit]
        return [{"name": t.name, "description": t.description or ""} for t in results]

    def get_details(self, name: str) -> dict[str, Any]:
        """Layer 2: return the full schema for exactly one tool."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"no such tool: {name}"}
        return mcp_tool_to_openai_param(tool)


# Meta-tools exposed to the model in progressive discovery mode. This array
# never changes shape across turns, which keeps the provider's prompt cache
# on the tools-prefix intact even as the *real* tool catalog grows.
META_TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "description": (
                "Search the full catalog of available tools by keyword. Returns "
                "matching tool names with a one-line description each. Call this "
                "before calling any tool you haven't inspected yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the capability you need",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tool_details",
            "description": (
                "Fetch the full input schema and documentation for one specific "
                "tool by name, so you know exactly what arguments to pass."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact tool name from search_tools results"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tool",
            "description": "Execute a tool by name with the given arguments and return its result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact tool name, as returned by search_tools"},
                    "arguments": {"type": "object", "description": "Arguments matching the tool's input schema"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
]


class ProgressiveDiscoveryAgent:
    """Drives an OpenAI chat completion against an MCP server's tools,
    choosing between naive (load-everything) and progressive discovery based
    on how much context the full tool catalog would consume.
    """

    def __init__(self, session: ClientSession, mode: str = "auto"):
        self.session = session
        self.mode = mode
        self.catalog = ToolCatalog()
        self.openai = OpenAI()

    async def bootstrap(self) -> str:
        result = await self.session.list_tools()
        self.catalog.refresh(result.tools, result.meta)
        return self._decide_mode()

    async def on_server_notification(self, notification) -> None:
        # Invalidate the cache immediately -- per the caching guidelines, a
        # list_changed notification makes the cache stale even if its TTL
        # hasn't lapsed yet. The actual re-fetch happens lazily, the next
        # time ensure_fresh() is called (see run()).
        from mcp.types import ToolListChangedNotification

        if isinstance(notification.root, ToolListChangedNotification):
            logger.info("Received tools/list_changed -- marking catalog stale")
            self.catalog.mark_stale()

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

    def _tools_param(self) -> list[ChatCompletionToolParam]:
        if self.active_mode == "direct":
            return [mcp_tool_to_openai_param(t) for t in self.catalog.tools.values()]
        return META_TOOLS

    async def _execute_meta_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "search_tools":
            await self.catalog.ensure_fresh(self.session)
            return self.catalog.search(arguments["query"])
        if name == "get_tool_details":
            await self.catalog.ensure_fresh(self.session)
            return self.catalog.get_details(arguments["name"])
        if name == "call_tool":
            return await self._call_real_tool(arguments["name"], arguments.get("arguments", {}))
        raise ValueError(f"unknown meta-tool: {name}")

    async def _call_real_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        logger.info("-> tools/call {} {}", name, arguments)
        result = await self.session.call_tool(name, arguments=arguments)
        if result.isError:
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return {"error": text}
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    async def run(self, user_message: str, max_turns: int = 8) -> tuple[str, int]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        tools_param = self._tools_param()
        total_tokens = 0

        for _ in range(max_turns):
            response = self.openai.chat.completions.create(
                model=MODEL,
                tools=tools_param,
                messages=messages,
            )
            if response.usage:
                total_tokens += response.usage.total_tokens
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or "", total_tokens

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")
                if self.active_mode == "progressive":
                    output = await self._execute_meta_tool(name, arguments)
                else:
                    output = await self._call_real_tool(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(output) if not isinstance(output, str) else output,
                    }
                )

        return "Reached max turns without a final answer.", total_tokens


async def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive tool discovery MCP client")
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
        default="progressive",
        help="auto (default): pick based on estimated token cost. direct: always send full schemas. "
        "progressive: always use search_tools/get_tool_details/call_tool.",
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
