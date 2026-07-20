def call_tool(action, tool_input):
    action = action.lower().strip()

    if action == "calculator":
        from tools.calculator_tool import run as calculator_tool

        return calculator_tool(tool_input)
    
    if action == "search":
        from tools.search_tool import run as search_tool

        return search_tool(tool_input)

    return f"unknown tool: {action}"