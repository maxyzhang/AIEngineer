# AI Engineer Agent Framework

A modular AI Agent framework that combines Retrieval-Augmented Generation (RAG), Reflection, Hybrid Search, and Tool Calling to answer technical and project-specific questions using a local knowledge base.

---

## Overview

This project demonstrates how to build an autonomous AI Agent capable of:

- Searching a local knowledge base
- Reasoning about retrieved information
- Deciding whether additional search is needed
- Generating grounded answers
- Supporting future tool execution and workflow automation

Unlike a traditional RAG chatbot, this project introduces an agentic workflow where the model evaluates retrieval quality before producing the final response.

---

## Architecture

```
                    User Question
                           │
                           ▼
                     Agent Controller
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
   Reflection Engine                 Tool Router (Future)
          │
          ▼
     Hybrid Retrieval
          │
     ┌────┴────┐
     │         │
     ▼         ▼
Vector Search  Keyword Search
     │         │
     └────┬────┘
          ▼
     ChromaDB Vector Store
          │
          ▼
 Local Knowledge Base (.txt)
          │
          ▼
     Grounded Response
```

---

## Features

### Reflection-based Agent

Instead of answering immediately, the agent first evaluates:

- Is the retrieved information sufficient?
- Should another search be performed?
- Is the answer trustworthy?

This reduces hallucinations and improves answer quality.

---

### Hybrid Search

The retrieval engine combines:

- Semantic Vector Search
- Keyword Search
- Confidence Scoring

This improves retrieval accuracy over pure vector search.

---

### Local Knowledge Base

The knowledge base is organized into modular domains:

```
knowledge/
│
├── companies/
├── projects/
│      ├── cdis/
│      ├── nvidia/
│      ├── linux_migration/
│      └── interview/
├── education/
├── skills/
├── resume.txt
└── project.txt
```

---

### Automatic Knowledge Indexing

Documents are automatically:

- Loaded
- Chunked
- Embedded
- Stored into ChromaDB

using Sentence Transformers.

---

## Technologies

- Python
- ChromaDB
- Sentence Transformers
- all-MiniLM-L6-v2
- NumPy
- Scikit-Learn
- Local RAG
- Reflection Pattern
- Agentic Workflow

---

## Project Structure

```
AIEngineer/

agent.py
agent_loop.py
build_vector_db.py
vector_search.py
tools.py
tool_agent.py
chat.py

knowledge/
chromadb_db/

README.md
requirements.txt
```

---

## Retrieval Workflow

```
Question
   │
   ▼
Embedding
   │
   ▼
Vector Search
   │
   ▼
Keyword Search
   │
   ▼
Merge Results
   │
   ▼
Reflection
   │
   ├── Search Again
   └── Answer
```

---

## Example

User:

```
Describe your NVIDIA CUDA experience.
```

Agent:

```
Reflection:
Decision: ANSWER

Reason:
Enough evidence has been retrieved from the NVIDIA project documentation.

Sources:
- projects/nvidia/cuda.txt
- projects/interview/nvidia_collaboration.txt

Answer:
...
```

---

## Current Capabilities

✅ Local RAG

✅ Reflection Loop

✅ Hybrid Retrieval

✅ Confidence Evaluation

✅ Automatic Knowledge Indexing

✅ Modular Knowledge Organization

---

## Planned Features

- Multi-step Tool Calling
- Memory Management
- Planner Agent
- Multi-Agent Collaboration
- Web Search Integration
- Cross-Encoder Re-ranking
- Query Expansion
- Metadata Filtering
- Conversation Memory
- LangGraph-style Workflow Engine

---

## Design Goals

- Reduce hallucinations
- Improve retrieval quality
- Explain agent reasoning
- Modular architecture
- Production-ready design
- Easy extension with new tools

---

## Author

Max Zhang

Senior Software Engineer

AI Infrastructure | HPC | RAG | Agentic AI | LLM Applications
