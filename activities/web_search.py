"""
Web Search over MCP

An MCP server offers tools to any client that speaks the protocol. You will
write a web search tool with Tavily and serve it over MCP. The bottom of the
file is a small Gemini agent that starts your server, lists its tools, and
calls them to answer a question.

    # needs Python 3.10 or newer
    pip install "mcp>=2" tavily-python google-genai
    export TAVILY_API_KEY=...       # free key at https://app.tavily.com
    export GEMINI_API_KEY=...
    python web_search.py
"""

import asyncio
import logging
import os
import sys

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import MCPServer
from tavily import TavilyClient

# The Gemini SDK logs a warning about "automatic function calling" on the first
# request, for a feature this file does not use.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

MODEL = "gemini-3.5-flash-lite"
QUESTION = "What was the news in NYC on October 25th, 2026?"
MAX_CHARS = 8000
MAX_STEPS = 6

mcp = MCPServer("web-search")
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# =============================================================================
# Server
# =============================================================================
# The agent sees only each tool's name, docstring, and argument types, and
# decides from those when to call it.
#
#   tools: https://py.sdk.modelcontextprotocol.io/v2/servers/tools/

# --- Task 1: Search ----------------------------------------------------------
# Tavily is a search API built for LLMs. It returns ranked results with the
# relevant text of each page already pulled out.
#
#   tavily: https://docs.tavily.com/sdk/python/reference

@mcp.tool()
def web_search(query: str) -> list[dict]:
    """TODO: what this tool does and when to use it"""
    # TODO: return the title, URL, and text of each result
    return []


# --- Task 2: Read a page -----------------------------------------------------
# A search result holds only part of its page.

@mcp.tool()
def read_page(url: str) -> str:
    """TODO: what this tool does and when to use it"""
    # TODO: return the text of the page, at most MAX_CHARS characters
    return ""


# --- Task 3: Run -------------------------------------------------------------
# Run the file. It prints every tool call the agent makes, then its answer.
# Then change QUESTION to one the model can answer without searching, and to
# one that needs a full page, and see which tools it calls.


# --- Task 4: Another client --------------------------------------------------
# Your server works with any MCP client. Add it to one, with the command
#
#   /full/path/to/python /full/path/to/web_search.py serve
#
# and TAVILY_API_KEY set in the server's environment, then ask it QUESTION.
#
#   gemini cli:  https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/mcp-server.md
#   claude code: https://code.claude.com/docs/en/mcp
#   vs code:     https://code.visualstudio.com/docs/copilot/chat/mcp-servers


# =============================================================================
# Client (written for you)
# =============================================================================

async def ask(question: str) -> str:
    """Starts this file as an MCP server and answers question with its tools."""
    server = StdioServerParameters(command=sys.executable,
                                   args=[__file__, "serve"], env=dict(os.environ))
    async with stdio_client(server) as (read, write), \
            ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))
        config = types.GenerateContentConfig(tools=[types.Tool(
            function_declarations=[types.FunctionDeclaration(
                name=t.name, description=t.description,
                parameters_json_schema=t.input_schema) for t in tools])])
        gemini = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=8)))
        contents = [types.Content(role="user", parts=[types.Part(text=question)])]
        for _ in range(MAX_STEPS):
            response = await gemini.aio.models.generate_content(
                model=MODEL, contents=contents, config=config)
            if not response.function_calls:
                return response.text or ""
            contents.append(response.candidates[0].content)
            results = []
            for call in response.function_calls:
                print(f"  {call.name}({call.args})")
                result = await session.call_tool(call.name, call.args or {})
                text = "\n".join(c.text for c in result.content if c.type == "text")
                results.append(types.Part.from_function_response(
                    name=call.name, response={"result": text}))
            contents.append(types.Content(role="user", parts=results))
        return f"(no answer after {MAX_STEPS} steps)"


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        mcp.run()
    else:
        print("\n" + asyncio.run(ask(QUESTION)))
