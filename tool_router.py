from tools.calculator_tool import run as calculator_tool
from tools.search_tool import run as search_tool

TOOLS = {
    "calculator": calculator_tool,
    "search": search_tool,
}

def call_tool(action, tool_input):
    tool = TOOLS.get(action.lower())
    if tool is None:
        return f"unknown tool: {action}"

    return tool(tool_input)