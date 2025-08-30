import asyncio
import json
import os
from typing import Optional
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI
import mcp.client.sse as _sse_mod
from httpx import AsyncClient as _BaseAsyncClient
from loguru import logger
import aiohttp

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

import httpx
_orig_request = httpx.AsyncClient.request

async def _patched_request(self, method, url, *args, **kwargs):
    # ensure follow_redirects is set so 307 → /messages/ works
    kwargs.setdefault("follow_redirects", True)
    return await _orig_request(self, method, url, *args, **kwargs)

httpx.AsyncClient.request = _patched_request
def llm_client(message: str):
    try:
        client = OpenAI(base_url='https://genai-api-dev.dell.com/v1',http_client=httpx.Client(verify=False),
    api_key=os.environ["DEV_GENAI_API_KEY"])

        completion = client.chat.completions.create(
            model="llama-3-1-8b-instruct",
            messages=[
                {"role": "system", "content": "You are an intelligent Assistant. You will execute tasks as instructed"},
                {
                    "role": "user",
                    "content": message,
                },
            ],
        )
        
        print(completion)

        result = completion.choices[0].message.content
        return result
    except Exception as e:
        logger.exception(e)



def get_prompt_to_identify_tool_and_arguements(query, tools):
    tools_description = "\n".join([f"{tool.name}: {tool.description}, {tool.inputSchema}" for tool in tools.tools])
    return  ("You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. \n"
                f"User's Question: {query}\n"                
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "                
                "the exact JSON object format below, nothing else:\n"
                "Keep the values in str "
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n")
    



TOKEN_URL = "http://localhost:8100/token"
SSE_URL = "http://localhost:8100/sse"

async def get_token():
    payload = {"client_id": "test_client", "client_secret": "secret_1234"}
    async with aiohttp.ClientSession() as session:
        async with session.post(TOKEN_URL, json=payload) as resp:
            if resp.status != 200:
                logger.error(f"Failed to get token: {resp.status}")
                raise Exception("Unable to authenticate. Ensure you are using valid credentials")
            data = await resp.json()
            logger.info("Successfully generated token")
            return data["access_token"]

async def main(query:str):        
    try:
        auth_token = await get_token()    
        headers = {"Authorization": f"Bearer {auth_token}"}
        async with sse_client(url=SSE_URL,headers=headers) as (in_stream, out_stream):
            # 2) Create an MCP session over those streams
            async with ClientSession(in_stream, out_stream) as session:
                # 3) Initialize
                info = await session.initialize()
                logger.info(f"Connected to {info.serverInfo.name} v{info.serverInfo.version}")

                # 4) List tools
                tools = (await session.list_tools())
                logger.info(tools)            
                
                prompt = get_prompt_to_identify_tool_and_arguements(query,tools)
                logger.info(f"Printing Prompt \n {prompt}")
                
                response = llm_client(prompt)
                print(response)
                
                tool_call = json.loads(response)
                            
                result = await session.call_tool(tool_call["tool"], arguments=tool_call["arguments"])
                logger.success(f"User query: {query}, Tool Response: {result.content[0].text}")
    except Exception as e:
        print(f"Encountered error: {e}")

            

if __name__ == "__main__":
    
    queries = ["What is the time in Bengaluru?", "What is the weather like right now in Dubai?"]
    for query in queries:
        asyncio.run(main(query))
