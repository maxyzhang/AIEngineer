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
    },
    {
        "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a local text file from the project folder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The local file path, for example README.md"
                        }
                    },
                    "required": ["file_path"]
                }
        }         
    }
]