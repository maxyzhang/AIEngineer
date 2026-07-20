import numpy as np
import math
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
from typing import Any
from openai_client import get_client

_memory_client: Any | None = None

MEMORY_FILE = "memory.json"
HISTORY_FILE = "history.json"  # agent step history
CONVERSATION_FILE = "conversation.json"  # user chat history
AUDIT_LOG_FILE = "memory_audit.jsonl"

memory_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_memory_client() -> Any:
    """Create the OpenAI client only when memory processing needs it."""

    global _memory_client

    if _memory_client is None:
        _memory_client = get_client()

    return _memory_client

def log_memory_event(
    event_type,
    memory_text,
    details=None,
):
    """
    Append one memory lifecycle event as a JSON line.
    """

    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "memory_text": str(memory_text).strip(),
        "details": details or {},
    }

    try:
        with open(
            AUDIT_LOG_FILE,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError as error:
        print(
            "[Memory Audit] Failed to write event:",
            error,
        )

def rank_memories(items):
    normalized_items = []

    for item in items:
        normalized = normalize_memory_item(item)

        if normalized["text"]:
            normalized_items.append(normalized)

    return sorted(
        normalized_items,
        key=lambda item: (
            item["importance"],
            item["access_count"],
            item["created_at"],
        ),
        reverse=True,
    )

def retrieve_memories(question, memory, limit=20):
    if not question or not question.strip():
        return []

    items = memory.get("long_term_memory", [])
    normalized_items = []

    for item in items:
        normalized = normalize_memory_item(item)

        if normalized["text"]:
            normalized_items.append(normalized)

    if not normalized_items:
        return []

    texts = [item["text"] for item in normalized_items]

    embeddings = memory_model.encode(
        [question] + texts,
        normalize_embeddings=True,
    )

    question_embedding = embeddings[0]
    memory_embeddings = embeddings[1:]

    similarities = memory_embeddings @ question_embedding

    ranked_pairs = sorted(
        zip(normalized_items, similarities),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    return [
        {
            **item,
            "similarity": float(similarity),
        }
        for item, similarity in ranked_pairs[:limit]
    ]

def calculate_recency_score(item, decay_days=30):
    """
    Return a score between 0 and 1 based on how recently
    the memory was accessed or created.
    """

    timestamp = (
        item.get("last_accessed")
        or item.get("created_at")
    )

    if not timestamp:
        return 0.0

    try:
        memory_time = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return 0.0

    age_seconds = max(
        (datetime.now() - memory_time).total_seconds(),
        0.0,
    )

    age_days = age_seconds / 86400.0

    return math.exp(-age_days / decay_days)

def consolidate_memories(
    items,
    similarity_threshold=0.85,
):
    """
    Remove semantically similar memories from a ranked result list.

    The input order is preserved, so the highest-ranked memory in each
    semantic group is retained.
    """

    if not items:
        return []

    normalized_items = []

    for item in items:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()

        if text:
            normalized_items.append(item)

    if len(normalized_items) <= 1:
        return normalized_items

    texts = [
        str(item.get("text", "")).strip()
        for item in normalized_items
    ]

    embeddings = memory_model.encode(
        texts,
        normalize_embeddings=True,
    )

    consolidated_items = []
    selected_embeddings = []

    for item, embedding in zip(normalized_items, embeddings):
        is_duplicate = False

        for selected_embedding in selected_embeddings:
            similarity = float(
                np.dot(embedding, selected_embedding)
            )

            if similarity >= similarity_threshold:
                is_duplicate = True

                print(
                    "[Memory Consolidation] Skipped similar memory: "
                    f"{item.get('text', '')}"
                )

                log_memory_event(
                    "consolidated",
                    item.get("text", ""),
                    {
                        "reason": "semantic_similarity",
                        "similarity_threshold": similarity_threshold,
                    },
                )

                break

        if not is_duplicate:
            consolidated_items.append(item)
            selected_embeddings.append(embedding)

    return consolidated_items

def get_relevant_memory_text(
    question,
    memory,
    retrieval_limit=20,
    final_limit=8,
    minimum_similarity=0.30,
):
    retrieved_items = retrieve_memories(
        question,
        memory,
        limit=retrieval_limit,
    )

    scored_items = []

    for item in retrieved_items:
        similarity = float(item.get("similarity", 0.0))
        importance = int(item.get("importance", 5))
        access_count = int(item.get("access_count", 0))

        if similarity < minimum_similarity:
            continue
        recency_score = calculate_recency_score(item)

        final_score = (
            similarity * 0.65
            + (importance / 10.0) * 0.15
            + min(access_count, 10) / 10.0 * 0.10
            + recency_score * 0.10
        )

        scored_items.append(
            {
                **item,
                "recency_score": recency_score,
                "final_score": final_score,
            }
        )

        print(
            f"[Memory Rank] "
            f"similarity={similarity:.3f}, "
            f"importance={importance}, "
            f"access={access_count}, "
            f"recency={recency_score:.3f}, "
            f"final={final_score:.3f}, "
            f"{item.get('text', '')}"
        )
        
    scored_items.sort(
        key=lambda item: item.get("final_score", 0.0),
        reverse=True,
    )

    consolidated_items = consolidate_memories(
        scored_items,
        similarity_threshold=0.85,
    )
    final_items = consolidated_items[:final_limit]

    return final_items

def format_final_memory_context(retrieved_items):
    """
    Convert retrieved memory items into a prompt string.
    """

    if not retrieved_items:
        return "No relevant long-term memory found."

    lines = []

    for item in retrieved_items:
        if isinstance(item, dict):
            text = item.get("text", "").strip()
        else:
            text = str(item).strip()

        if text:
            lines.append(f"- {text}")

    return "\n".join(lines)

def reinforce_memories(memory, retrieved_items):
   if not retrieved_items:
       return memory

   now = datetime.now().isoformat()
   stored_items = memory.get("long_term_memory", [])

   retrieved_texts = {
       str(item.get("text", "")).strip().lower()
       for item in retrieved_items
       if isinstance(item, dict) and item.get("text")
   }

   for stored_item in stored_items:
       if not isinstance(stored_item, dict):
           continue

       stored_text = str(stored_item.get("text", "")).strip().lower()

       if stored_text in retrieved_texts:
           stored_item["access_count"] = int(
               stored_item.get("access_count", 0)
           ) + 1
        
           stored_item["last_accessed"] = now

           log_memory_event(
               "reinforced",
               stored_item.get("text", ""),
               {
                   "access_count": stored_item["access_count"],
                   "importance": stored_item.get("importance", 5),
                   "last_accessed": stored_item["last_accessed"],
               },
           )
          
           current_importance = int(
               stored_item.get("importance", 5)
           )

           if (
                stored_item["access_count"] % 5 == 0
                and current_importance < 10
           ):
               old_importance = current_importance
               stored_item["importance"] = current_importance + 1

               log_memory_event(
                   "importance_increased",
                   stored_item.get("text", ""),
                   {
                        "old_importance": old_importance,
                        "new_importance": stored_item["importance"],
                        "access_count": stored_item["access_count"],
                   },
               )
               print(
                   "[Memory] Importance increased "
                   f"to {stored_item['importance']}: "
                   f"{stored_item.get('text', '')}"
               )
               
           print(
               f"[Memory] Reinforced "
               f"access_count={stored_item['access_count']}: "
               f"{stored_item.get('text', '')}"
           )

   memory["long_term_memory"] = stored_items
   return memory

def decay_memories(
    memory,
    decay_interval_days=30,
    minimum_importance=1,
):
    """
    Reduce importance for memories that have not been accessed recently.

    One importance point is removed for each complete decay interval.
    Memories are never reduced below minimum_importance.
    """

    now = datetime.now()
    stored_items = memory.get("long_term_memory", [])

    for item in stored_items:
        if not isinstance(item, dict):
            continue

        timestamps = (
            item.get("last_decayed_at"),
            item.get("last_accessed"),
            item.get("created_at"),
        )

        parsed_times = []

        for timestamp in timestamps:
            if not timestamp:
                continue

            try:
                parsed_times.append(datetime.fromisoformat(timestamp))
            except (TypeError, ValueError):
                continue
        if not parsed_times:
            continue

        if not parsed_times:
            continue

        try:
            reference_time = max(parsed_times)
        except (TypeError, ValueError):
            continue

        age_days = max(
            (now - reference_time).total_seconds() / 86400.0,
            0.0,
        )

        decay_steps = int(age_days // decay_interval_days)

        if decay_steps <= 0:
            continue

        current_importance = int(item.get("importance", 5))
        decayed_importance = max(
            minimum_importance,
            current_importance - decay_steps,
        )

        if decayed_importance < current_importance:
            item["importance"] = decayed_importance
            item["last_decayed_at"] = now.isoformat()

            print(
                "[Memory Decay] "
                f"importance {current_importance}"
                f" -> {decayed_importance}: "
                f"{item.get('text', '')}"
            )

            log_memory_event(
                "decayed",
                item.get("text", ""),
                {
                    "old_importance": current_importance,
                    "new_importance": decayed_importance,
                    "decay_steps": decay_steps,
                    "age_days": round(age_days, 2),
                    "last_decayed_at": item["last_decayed_at"],
                },
            )

    memory["long_term_memory"] = stored_items
    return memory

def garbage_collect_memories(
    memory,
    minimum_age_days=180,
    maximum_importance=1,
    maximum_access_count=0,
):
    """
    Remove memories only when they are old, low-importance,
    and unused.

    A memory is deleted only when all conditions are met:
    - importance <= maximum_importance
    - access_count <= maximum_access_count
    - inactive for at least minimum_age_days
    """

    now = datetime.now()
    stored_items = memory.get("long_term_memory", [])
    retained_items = []

    for item in stored_items:
        if not isinstance(item, dict):
            retained_items.append(item)
            continue

        importance = int(item.get("importance", 5))
        access_count = int(item.get("access_count", 0))

        reference_timestamp = (
            item.get("last_accessed")
            or item.get("created_at")
        )

        if not reference_timestamp:
            retained_items.append(item)
            continue

        try:
            reference_time = datetime.fromisoformat(
                reference_timestamp
            )
        except (TypeError, ValueError):
            retained_items.append(item)
            continue

        age_days = max(
            (now - reference_time).total_seconds() / 86400.0,
            0.0,
        )

        should_delete = (
            importance <= maximum_importance
            and access_count <= maximum_access_count
            and age_days >= minimum_age_days
        )

        if should_delete:
            print(
                "[Memory GC] Removed stale memory: "
                f"{item.get('text', '')}"
            )

            log_memory_event(
                "garbage_collected",
                item.ge("text", ""),
                {
                    "importance": importance,
                    "access_count": access_count,
                    "age_days": round(age_days, 2),
                },
            )
            continue

        retained_items.append(item)

    memory["long_term_memory"] = retained_items
    return memory

def normalize_memory_item(item):
    now = datetime.now().isoformat()

    if isinstance(item, dict):
        return {
            "text": item.get("text", "").strip(),
            "importance": int(item.get("importance", 5)),
            "access_count": int(item.get("access_count", 0)),
            "created_at": item.get("created_at", now),
            "last_accessed": item.get("last_accessed", ""),
            "last_decayed_at": item.get("last_decayed_at",""),
        }

    return {
        "text": str(item).strip(),
        "importance": 5,
        "access_count": 0,
        "created_at": now,
        "last_accessed": None,
        "last_decayed_at": item.get("last_decayed_at",""),
    }

def clear_conversation_memory():
    save_history([])

def clear_history():
    save_history([])

def load_conversation():
    if not os.path.exists(CONVERSATION_FILE):
        return []
    
    with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_conversation(conversation):
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

def get_memory_text(memory, limit=10):
    items = memory.get("long_term_memory", [])
    ranked_items = rank_memories(items)[:limit]
    
    return "\n".join(
        f"= {item['text']}"
        for item in ranked_items
    )

def extract_memory(question, answer):
    prompt = f"""
You are a memory extraction system.

Extract only durable long-term facts that should be remembered about the user.

Good memories:
- User projects
- User role
- Skills
- Preferences
- Long-term goals
- Important ongoing work

Do not store:
- Temporary questions
- One-time calculations
- Generic assistant answers
- Duplicates

User question:
{question}

Assistant answer:
{answer}

Assign an importance score to every memory.

10 = User identity
9 = Current long-term project
8 = Long-tem goal or profession
7 = Strong preference
6 = Stable skill or knowledge
5 = General background fact

If thr memory describes the user's CURRENT project or product, always assign importance 9.

If a memory describes the user's profession or long-term career, assign importance 8.

Do not assign importance 5 to a current long-term project.

Return ONLY JSON in this format:
{{
  "facts": [
    {{
        "text": "fact 1",
        "importance": 8
    }},
    {{
        "text": "fact 2,
        "importance": 6
    }}
  ]
}}
"""

    response = get_memory_client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()

    try:
        data = json.loads(text)
        return data.get("facts", [])
    except Exception:
        return []

def cosine_similarity(vector_a, vector_b):
    vetcor_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)

def is_duplicate_memory(
    new_fact,
    existing_facts,
    similarity_threshold=0.88,
):
    if not new_fact or not existing_facts:
        return False

    new_fact_normalized = new_fact.strip().lower()

    # 先做完全相同检查，避免不必要的 embedding
    for existing_fact in existing_facts:
        existing_text = (
            existing_fact.get("text", "")
            if isinstance(existing_fact, dict)
            else str(existing_fact)
        )

        if new_fact_normalized == existing_text.strip().lower():
            return True

    existing_texts = [
        item.get("text", "")
        if isinstance(item, dict)
        else str(item)
        for item in existing_facts
    ]

    existing_texts = [
        text.strip()
        for text in existing_texts
        if text and text.strip()
    ]

    if not existing_texts:
        return False

    embeddings = memory_model.encode(
        [new_fact] + existing_texts,
        normalize_embeddings=True,
    )

    new_embedding = embeddings[0]

    for existing_embedding in embeddings[1:]:
        similarity = float(
            np.dot(new_embedding, existing_embedding)
        )

        if similarity >= similarity_threshold:
            return True

    return False

def merge_memory(
    memory,
    new_facts,
    similarity_threshold=0.8,
):
    if "long_term_memory" not in memory:
        memory["long_term_memory"] = []

    # 先把旧字符串格式统一转换成带 metadata 的 dict
    existing_facts = [
        normalize_memory_item(item)
        for item in memory["long_term_memory"]
    ]

    bad_phrases = [
        "i don't have",
        "not enough information",
        "available observations",
        "appears to",
        "not yet known",
        "cannot confirm",
        "no evidence",
    ]

    for item in new_facts:
        # 兼容旧版字符串和新版 dict
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()

            try:
                importance = int(item.get("importance", 5))
            except (TypeError, ValueError):
                importance = 5
        else:
            text = str(item).strip()
            importance = 5

        if not text:
            continue

        # 限制在 1–10
        importance = max(1, min(10, importance))

        text_lower = text.lower()

        if any(phrase in text_lower for phrase in bad_phrases):
            print(f"[Memory] Skipped low-value memory: {text}")
            continue

        if is_duplicate_memory(
            text,
            existing_facts,
            similarity_threshold,
        ):
            print(f"[Memory] Skipped duplicate: {text}")
            continue

        new_item = {
            "text": text,
            "importance": importance,
            "access_count": 0,
            "created_at": datetime.now().isoformat(),
            "last_accessed": None,
        }

        existing_facts.append(new_item)
        print(
            f"[Memory] Added importance={importance}: {text}"
        )

    memory["long_term_memory"] = existing_facts
    return memory

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "last_question": "",
            "last_answer": "",
            "long_term_memory": [],
            "updated_at": ""
        }

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    memory["updated_at"] = datetime.now().isoformat()

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_memory_item(text):
    memory = load_memory()

    if "long_term_memory" not in memory:
        memory["long_term_memory"] = []

    memory["long_term_memory"].append({
        "text": text,
        "created_at": datetime.now().isoformat()
    })

    save_memory(memory)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_conversation():
    if not os.path.exists(CONVERSATION_FILE):
        return []

    with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_conversation(conversation):
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)


def add_conversation_turn(question, answer):
    conversation = load_conversation()

    conversation.append({
        "question": question,
        "answer": answer,
        "created_at": datetime.now().isoformat()
    })

    save_conversation(conversation)


def get_conversation_context(limit=5):
    conversation = load_conversation()
    recent = conversation[-limit:]

    return "\n".join(
        f"User: {item.get('question', '')}\nAssistant: {item.get('answer', '')}"
        for item in recent
    )