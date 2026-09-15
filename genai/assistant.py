import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment
load_dotenv()
is_mock_mode = os.getenv("MOCK_LLM") == "True"

# --- 1. Define the AI's Memory (State) ---
class AgentState(TypedDict):
    question: str
    context: str
    answer: str

# --- 2. Define the Steps (Nodes) ---
def retrieve_node(state: AgentState):
    print("[System] Retrieving data...")
    # In a real app, this searches your SQLite DB or CSV. 
    mock_context = "Titanic data shows 1st class passengers had much higher survival rates."
    return {"context": mock_context}

def generate_node(state: AgentState):
    print("[System] Generating response...")
    if is_mock_mode:
        # Our free, safe testing response
        answer = f"MOCK AI RESPONSE: Based on the data ({state['context']}), class heavily impacted survival."
    else:
        answer = "Real OpenAI/Gemini API response would go here."
    return {"answer": answer}

# --- 3. Build the Flowchart (Graph) ---
workflow = StateGraph(AgentState)

# Add our steps
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Connect them together
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the assistant
app = workflow.compile()

# --- 4. Run the Assistant ---
if __name__ == "__main__":
    print("--- GenAI Assistant Initialized ---")
    test_question = "Did passenger class affect Titanic survival?"
    print(f"User: {test_question}\n")
    
    # Run the flowchart
    result = app.invoke({"question": test_question})
    
    print("\n--- Final Output ---")
    print(result["answer"])