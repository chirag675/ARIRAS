import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.vectorstore import load_vectorstore
from core.edge_handler import score_rag_response, rag_fallback_response  # ← NEW
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are ARIRAS — an expert AI regulatory compliance analyst.
Use ONLY the regulation text below to answer the question.
Be precise, cite article numbers or clause references where available.
If the answer is not in the context, say: "This information is not found in the uploaded regulation."

REGULATION CONTEXT:
{context}

QUESTION: {question}

ANSWER (be specific, reference clauses):
""",
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask_regulation(question: str) -> dict:
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs        = retriever.invoke(question)

    # Hard block — nothing retrieved
    if not docs:
        return rag_fallback_response(
            "No regulation content matched your question. "
            "Ensure a regulation PDF is indexed, then rephrase."
        )

    context = format_docs(docs)
    answer  = (RAG_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )

    sources = [
        f"[Page {d.metadata.get('page', '?')}] "
        f"{d.page_content[:300].strip().replace(chr(10), ' ')}..."
        for d in docs
    ]

    # Two-layer confidence scoring (embedding + LLM judge)
    edge = score_rag_response(answer, docs, question)

    return {"answer": answer, "sources": sources, "_edge": edge}