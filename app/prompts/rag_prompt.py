def build_rag_prompt(context: str, history: str, question: str) -> str:
    return f"""You are a helpful support assistant.

Use ONLY the provided context to answer the user. If the context does not contain enough information, say that you could not find enough information in the knowledge base. Keep the answer concise and practical.

Context:
{context}

Conversation History:
{history}

Question:
{question}

Answer:"""
