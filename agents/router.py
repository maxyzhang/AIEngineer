def route(question):
    q = question.lower()

    if any(word in q for word in [
        "interview",
        "tell me about yourself",
        "leadership",
        "challenge",
        "weakness",
        "strength",
        "behavior"
    ]):
        return "interview"

    if any(word in q for word in [
        "resume",
        "cv",
        "job description",
        "jd",
        "tailor"
    ]):
        return "resume"

    return "knowledge"