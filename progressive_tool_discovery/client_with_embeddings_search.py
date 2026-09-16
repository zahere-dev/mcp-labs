"""
MCP client demonstrating the "progressive tool discovery" pattern described in
https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices

This variant uses EMBEDDING-based vector similarity, backed by a local
Chroma vector database, as the Layer 1 search strategy (see
client_with_keyword_search.py for the plain keyword variant). Embeddings
handle synonyms and semantic matching that keyword search misses -- e.g. a
query like "reach out to the customer" should still surface an `email_send`
tool even though none of those words appear in its description.

Naive host behaviour is: fetch tools/list from every connected server and hand
every full tool schema to the model at the start of the conversation. That's
fine for a handful of tools, but wasteful once schemas eat a meaningful slice
of the context window.

This client instead:

  1. Connects to the server and calls tools/list as normal, but caches the
     full definitions host-side instead of injecting them into the model's
     context (`ToolCatalog`).
  2. Indexes every tool's "name: description" text into an in-memory Chroma
     collection once at startup (and again whenever the catalog is
     refreshed). Chroma calls OpenAI's embeddings API under the hood (via
     `OpenAIEmbeddingFunction`) and handles vector storage + similarity
     search for us.
  3. Estimates how many tokens the full tool set would cost. If that is below
     a configurable percentage of the model's context window, it falls back
     to the naive "load everything" behaviour (small tool sets don't need
     progressive discovery). Otherwise it switches to progressive discovery.
  4. In progressive discovery mode, only three lightweight meta-tools are
     exposed to the model:
       - search_tools(query)      -> Layer 1: query the Chroma collection
       - get_tool_details(name)   -> Layer 2: inspect full schema on demand
       - call_tool(name, args)    -> Layer 3: execute
     The model never sees the real tool schemas in its `tools` array; it
     only sees them as JSON text returned by get_tool_details. Because the
     `tools` array handed to the API never changes shape, prompt caching of
     the tool-definitions prefix stays intact across turns.
  5. The catalog honors the server's `ttlMs` / `cacheScope` cache hints
     (2026-07-28 spec, carried in tools/list's `_meta`): it re-fetches and
     rebuilds the Chroma collection once the TTL lapses, and is marked stale
     immediately on a notifications/tools/list_changed notification even if
     the TTL hasn't expired yet -- the actual rebuild happens lazily on next
     use.

Run the server first:
    python server.py

Then:
    python client_with_embeddings_search.py "reach out to the customer about their open ticket"
    python client_with_embeddings_search.py --mode direct "what's the weather in Paris?"
    python client_with_embeddings_search.py --mode progressive "close ISSUE-1 with resolution 'fixed'"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool as MCPTool
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

load_dotenv()

SERVER_URL = "http://127.0.0.1:8300/mcp"
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_COLLECTION_NAME = "mcp-tool-catalog"
CONTEXT_WINDOW_TOKENS = 200_000
# Percentage of the context window that the *full* tool catalog is allowed to
# consume before we switch from "dump everything" to progressive discovery.
PROGRESSIVE_DISCOVERY_THRESHOLD_PCT = 1.0
# Chroma returns cosine *distance* (1 - similarity); below this similarity,
# a tool isn't considered a real match.
SIMILARITY_FLOOR = 0.2


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


class ToolCatalog:
    """Host-side cache of every tool a connected server exposes, backed by an
    in-memory Chroma collection that embeds each tool's name+description.

    This is populated from tools/list and re-populated on list_changed
    notifications. It is never sent to the model wholesale -- the model only
    ever sees what search_tools / get_tool_details choose to reveal.

    Per the 2026-07-28 spec, tools/list responses can carry `ttlMs` /
    `cacheScope` hints in `_meta` (see server.py). is_stale()/mark_stale()
    track those hints; ProgressiveDiscoveryAgent.ensure_fresh() does the
    actual re-fetch + re-index, since indexing is only needed in
    progressive mode.
    """

    def __init__(self) -> None:
        self.tools: dict[str, MCPTool] = {}
        self.ttl_ms: int | None = None
        self.cache_scope: str | None = None
        self.fetched_at: float = 0.0
        self.stale: bool = True
        self._chroma_client = chromadb.EphemeralClient()
        self._embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=EMBEDDING_MODEL,
        )
        self._collection: chromadb.Collection | None = None

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

    async def index(self) -> None:
        """(Re)build the Chroma collection from the current tool set. Runs
        the blocking Chroma/OpenAI calls in a thread so they don't stall the
        event loop. Chroma embeds the documents for us on `add`.
        """
        await asyncio.to_thread(self._index_sync)
        logger.info("Indexed {} tools into Chroma collection '{}'", len(self.tools), CHROMA_COLLECTION_NAME)

    def _index_sync(self) -> None:
        try:
            self._chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet -- nothing to clear
        self._collection = self._chroma_client.create_collection(
            CHROMA_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        if not self.tools:
            return
        names = list(self.tools.keys())
        documents = [f"{name}: {self.tools[name].description or ''}" for name in names]
        self._collection.add(ids=names, documents=documents)

    def full_definitions_token_estimate(self) -> int:
        blob = json.dumps([mcp_tool_to_openai_param(t) for t in self.tools.values()])
        return estimate_tokens(blob)

    async def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Layer 1: query the Chroma collection for the tools whose
        name+description embedding is closest to the query embedding.

        This is the "Embedding-based" strategy from the "Choosing a Discovery
        Strategy" section of the article -- it catches semantic matches that
        plain keyword search misses (synonyms, paraphrases), at the cost of
        an embedding call per search_tools invocation.
        """
        if self._collection is None or not self.tools:
            return [{"name": t.name, "description": t.description or ""} for t in list(self.tools.values())[:limit]]

        n_results = min(limit, len(self.tools))
        result = await asyncio.to_thread(
            self._collection.query, query_texts=[query], n_results=n_results
        )
        names = result["ids"][0]
        distances = result["distances"][0]

        # Chroma returns cosine *distance* (1 - similarity) when the
        # collection is configured with hnsw:space=cosine.
        matches = [(name, 1 - distance) for name, distance in zip(names, distances)]
        results = [name for name, similarity in matches if similarity >= SIMILARITY_FLOOR]
        if not results:
            # graceful fallback: nothing cleared the floor, surface the
            # closest matches anyway rather than leaving the model empty-handed.
            results = names

        return [{"name": name, "description": self.tools[name].description or ""} for name in results]

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
                "Search the full catalog of available tools by semantic similarity. "
                "Returns matching tool names with a one-line description each. Call "
                "this before calling any tool you haven't inspected yet."
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
        self.openai = OpenAI()
        self.catalog = ToolCatalog()

    async def bootstrap(self) -> str:
        result = await self.session.list_tools()
        self.catalog.refresh(result.tools, result.meta)
        active_mode = self._decide_mode()
        if active_mode == "progressive":
            await self.catalog.index()
        return active_mode

    async def on_server_notification(self, notification) -> None:
        # Invalidate the cache immediately -- per the caching guidelines, a
        # list_changed notification makes the cache stale even if its TTL
        # hasn't lapsed yet. The actual re-fetch + re-index happens lazily,
        # the next time ensure_fresh() is called (see _execute_meta_tool).
        from mcp.types import ToolListChangedNotification

        if isinstance(notification.root, ToolListChangedNotification):
            logger.info("Received tools/list_changed -- marking catalog stale")
            self.catalog.mark_stale()

    async def ensure_fresh(self) -> None:
        if not self.catalog.is_stale():
            return
        logger.info("Tool catalog cache expired or invalidated -- re-fetching tools/list")
        result = await self.session.list_tools()
        self.catalog.refresh(result.tools, result.meta)
        if self.active_mode == "progressive":
            await self.catalog.index()

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
            await self.ensure_fresh()
            return await self.catalog.search(arguments["query"])
        if name == "get_tool_details":
            await self.ensure_fresh()
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
    parser = argparse.ArgumentParser(description="Progressive tool discovery MCP client (embeddings search)")
    default_query = "reach out to the customer about their open ticket"
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
        help="auto (default): pick based on estimated token cost. direct: always send full schemas. "
        "progressive: always use search_tools/get_tool_details/call_tool.",
    )
    args = parser.parse_args()

    async with streamablehttp_client(SERVER_URL) as (read_stream, write_stream, _get_session_id):
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
