# AIEngineer

A learning project for building AI engineering application with Python and OpenAI API.

## Features

- OpenAI API client
- Command-line chatbot
- Conversation history
- Local JSON memory
- System prompt
- Modular project structure

## Project Structure

```text
AIEngineer/
├── app.py
├── chat.py
├── memory.py
├── openai_client.py
├── prompts.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python chat.py
```
