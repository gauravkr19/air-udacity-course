from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model = "openai/gpt-oss-120b",
    temperature=0.7,
    api_key=str(os.getenv("API_KEY")),
    base_url=str(os.getenv("BASE_URL"))
)

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    if location.lower() == "san francisco":
        return "Sunny, 72°F"
    return f"Weather data for {location} is unavailable."

# Bind tool(s) to model
llm_with_tools = llm.bind_tools([get_weather])

# Ask a question
messages = [HumanMessage(content="What's the weather in San Francisco?")]

# First model call: model may decide to call the tool
ai_message = llm_with_tools.invoke(messages)

print("AI tool calls:", ai_message.tool_calls)

# If the model asked for a tool, run it
if ai_message.tool_calls:
    tool_call = ai_message.tool_calls[0]

    # Actually execute the tool
    tool_result = get_weather.invoke(tool_call["args"])

    # Send the tool result back to the model
    tool_message = ToolMessage(
        content=tool_result,
        tool_call_id=tool_call["id"],
    )

    # Second model call: model uses the tool result to answer
    final_response = llm_with_tools.invoke(messages + [ai_message, tool_message])

    print("Final answer:", final_response.content)
else:
    print("Model answered directly:", ai_message.content)
