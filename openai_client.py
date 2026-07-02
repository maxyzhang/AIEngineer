from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

def get_client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )