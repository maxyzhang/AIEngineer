def run(expression):
    try:
        allowed = "0123456789+-*/(). "
        if not all(ch in allowed for ch in expression):
            return "Invalid expression."

        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Calculator error: {e}"