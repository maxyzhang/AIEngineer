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
    },
     {
        "type": "function",
            "function": {
                "name": "save_user_memory",
                "description": "Save a piece of log-term user memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                        },
                        "value": {
                            "type": "string"
                        }
                    },
                    "required": ["key", "value"]
                }
        }         
    },
    {
        "type": "function",
            "function": {
                "name": "get_user_memory",
                "description": "Get all long-term user memory.",
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
                "name": "search_knowledge",
                "description": "Search local knowledge files to answer questions about Max, his resume, company, project, or notes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question":{
                            "type": "string",
                            "description": "The user's question to search for in the knowledge base."
                        }
                    },
                    "required": ["question"]
                }
        }    
    }
]