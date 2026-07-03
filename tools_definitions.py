TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a math expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to calculate, for example; 135*27"
                        }
                    },
                    "required": ["expression"]
                }
            }    
    }
]