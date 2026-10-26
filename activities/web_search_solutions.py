# Web Search over MCP - SOLUTION KEY.

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

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

MODEL = "gemini-3.5-flash-lite"
QUESTION = "What was the news in NYC on October 25th, 2026?"
MAX_CHARS = 8000
MAX_STEPS = 6

mcp = MCPServer("web-search")
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# --- Task 1 ------------------------------------------------------------------

@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Search the web. Returns the title, URL, and most relevant text of each
    result. Use it for recent events or facts you are not sure of."""
    response = tavily.search(query, max_results=5)
    return [{"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response["results"]]


# --- Task 2 ------------------------------------------------------------------

@mcp.tool()
def read_page(url: str) -> str:
    """Return the text of the web page at a URL, such as one from web_search.
    Use it when a search result's text is not enough."""
    response = tavily.extract(url)
    if not response["results"]:
        return f"could not read {url}"
    return response["results"][0]["raw_content"][:MAX_CHARS]


# --- Task 4 ------------------------------------------------------------------
# Gemini CLI, in ~/.gemini/settings.json:
#
#   {"mcpServers": {"web-search": {
#       "command": "/full/path/to/python",
#       "args": ["/full/path/to/web_search_solutions.py", "serve"],
#       "env": {"TAVILY_API_KEY": "tvly-..."}}}}
#
# Claude Code:
#
#   claude mcp add --transport stdio --env TAVILY_API_KEY=tvly-... web-search \
#       -- /full/path/to/python /full/path/to/web_search_solutions.py serve


# --- Client ------------------------------------------------------------------

async def ask(question: str) -> str:
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
