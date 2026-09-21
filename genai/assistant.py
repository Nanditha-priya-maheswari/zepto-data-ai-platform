import os
from typing import TypedDict, List
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
import uvicorn

# 1. Load Environment & Setup Mock
load_dotenv()
is_mock_mode = os.getenv("MOCK_LLM") == "True"

# 2. Ingest 8 Policy Documents
policy_texts = {
    "Delivery": "We deliver groceries in 10 minutes within active geofences.",
    "Returns & Refunds": "Returns accepted within 24 hours for defective items. Refunds process in 3-5 days.",
    "Membership": "Zepto Pass offers free delivery on orders above 99 INR.",
    "Order Tracking": "Track your order in real-time through the app map interface.",
    "Cancellation": "Orders can only be cancelled before the packer begins processing.",
    "Damaged / Missing Items": "Report missing items immediately via in-app support for instant credit.",
    "Gift Cards": "Gift cards are non-refundable and valid for 1 year.",
    "Support Hours": "Customer support is available 24/7 via in-app chat."
}
docs = [Document(page_content=content, metadata={"source": title}) for title, content in policy_texts.items()]

# 3. Embeddings & Vector DB (ChromaDB + sentence-transformers)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Top-3 chunks via Cosine Similarity

# 4. LangGraph State & Nodes
class AgentState(TypedDict):
    question: str
    intent: str
    context: List[Document]
    answer: str
    sources: List[str]
    confidence: float

def classify_intent(state: AgentState):
    # Basic intent classifier for demonstration
    q = state["question"].lower()
    if any(keyword in q for keyword in ["delivery", "return", "refund", "pass", "cancel", "damaged", "gift", "support"]):
        return {"intent": "policy"}
    return {"intent": "general"}

def retrieve_and_answer(state: AgentState):
    # Policy Path
    retrieved_docs = retriever.invoke(state["question"])
    context_str = " ".join([d.page_content for d in retrieved_docs])
    sources = [d.metadata["source"] for d in retrieved_docs]
    
    if is_mock_mode:
        answer = f"[MOCK_LLM]: Based on policy: {context_str}"
    else:
        answer = "Real LLM Answer based on context."
        
    return {"context": retrieved_docs, "answer": answer, "sources": sources, "confidence": 0.95}

def direct_answer(state: AgentState):
    # General Path
    if is_mock_mode:
        answer = "[MOCK_LLM]: I am a Zepto support assistant. How can I help you today?"
    else:
        answer = "Real LLM general response."
        
    return {"context": [], "answer": answer, "sources": [], "confidence": 0.50}

# 5. Build LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify")

def route_intent(state: AgentState):
    return "retrieve_and_answer" if state["intent"] == "policy" else "direct_answer"

workflow.add_conditional_edges("classify", route_intent)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)
app_flow = workflow.compile()

# 6. FastAPI Setup
app = FastAPI(title="Zepto GenAI Assistant API")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    result = app_flow.invoke({"question": query.question, "intent": "", "context": [], "answer": "", "sources": [], "confidence": 0.0})
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)