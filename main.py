import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────
class EmailState(TypedDict):
    topic: str
    draft: str
    feedback: Optional[str]
    approved: bool
    iteration: int

# ── LLM ────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

# ── Nodes ──────────────────────────────────────────────────────────────────
def draft_email(state: EmailState) -> dict:
    topic = state["topic"]
    feedback = state.get("feedback")
    iteration = state.get("iteration", 0)

    if feedback:
        prompt = (
            f"You previously drafted an email about: '{topic}'.\n"
            f"The reviewer rejected it with this feedback: '{feedback}'.\n"
            f"Please rewrite the email addressing the feedback. "
            f"Keep it professional and concise."
        )
    else:
        prompt = (
            f"Draft a short, professional email about the following topic: '{topic}'.\n"
            f"Keep it concise (3-5 sentences)."
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "draft": response.content,
        "iteration": iteration + 1,
        "approved": False,
    }


def human_review(state: EmailState) -> dict:
    return state


# ── Routing ────────────────────────────────────────────────────────────────
def route_after_review(state: EmailState) -> str:
    if state.get("approved", False):
        return END
    return "draft_email"


# ── Build Graph ────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(EmailState)

    builder.add_node("draft_email", draft_email)
    builder.add_node("human_review", human_review)

    builder.set_entry_point("draft_email")
    builder.add_edge("draft_email", "human_review")
    builder.add_conditional_edges("human_review", route_after_review)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_review"]
    )


# ── CLI Runner ─────────────────────────────────────────────────────────────
def run():
    graph = build_graph()
    config = {"configurable": {"thread_id": "email-session-1"}}

    topic = input("📧 What should the email be about? ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    initial_state: EmailState = {
        "topic": topic,
        "draft": "",
        "feedback": None,
        "approved": False,
        "iteration": 0,
    }

    print("\n⚙️  Drafting email...\n")
    graph.invoke(initial_state, config)

    while True:
        snapshot = graph.get_state(config)
        current = snapshot.values

        iteration = current.get("iteration", 1)
        draft = current.get("draft", "")

        print(f"{'─'*60}")
        print(f"📝  Draft #{iteration}:\n")
        print(draft)
        print(f"{'─'*60}\n")

        decision = input("✅ Approve or ❌ Reject? [a/r]: ").strip().lower()

        if decision == "a":
            graph.update_state(config, {"approved": True, "feedback": None})
            graph.invoke(None, config)
            print("\n✅ Email approved! Final draft saved above.")
            break

        elif decision == "r":
            feedback = input("💬 Enter your feedback for redrafting: ").strip()
            if not feedback:
                print("Feedback cannot be empty for rejection.")
                continue

            graph.update_state(config, {"approved": False, "feedback": feedback})
            print("\n⚙️  Redrafting based on your feedback...\n")
            graph.invoke(None, config)

        else:
            print("Invalid input. Type 'a' to approve or 'r' to reject.")


if __name__ == "__main__":
    run()