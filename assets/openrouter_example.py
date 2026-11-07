"""pip install openai  |  export OPENROUTER_API_KEY="sk-or-..." """

import json
import os

from openai import OpenAI

MODEL = "qwen/qwen3-coder"  # any tool-calling model slug from https://openrouter.ai/models

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


# The tool: a normal Python function (stubbed here).
def get_weather(city):
    return {"city": city, "temp_f": 68, "conditions": "sunny"}


# The description the model sees (it never sees the code above).
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def run_agent():
    messages = [{"role": "user", "content": "What's the weather in Lisbon?"}]

    # 1. The model decides it needs the tool and asks for it.
    reply = client.chat.completions.create(model=MODEL, messages=messages, tools=tools).choices[0].message
    call = reply.tool_calls[0]

    # 2. We run the tool ourselves.
    result = get_weather(**json.loads(call.function.arguments))

    # 3. We send the result back and the model writes the answer.
    messages.append(reply.model_dump(exclude_none=True))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    print(final.choices[0].message.content)


run_agent()