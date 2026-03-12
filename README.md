# HITL Email Drafting Agent

A **Human-in-the-Loop (HITL)** email drafting application built with **LangGraph**.  
The AI drafts an email based on a given topic. The human can approve it or reject it with feedback.  
If rejected, the AI redrafts the email incorporating the feedback. This loop continues until the human approves.

---

## What is HITL?

Human-in-the-Loop (HITL) means a human is kept in the decision-making process of an AI workflow.  
Instead of the AI running fully autonomously, it **pauses and waits for human input** before proceeding.  
This is critical in real-world AI applications where accuracy, tone, and context matter.

---

## Graph Architecture
```
[START]
   ↓
draft_email          
   ↓
human_review    ← ⛔ graph pauses here (interrupt_before)
   ↓
approved?
   ├── YES ──→ [END]
   └── NO  ──→ draft_email (loops back with feedback)
```

### Nodes

| Node | Role |
|------|------|
| `draft_email` | LLM drafts or redrafts the email based on topic and feedback |
| `human_review` | Pause point — human approves or rejects with feedback |

### Key LangGraph Concepts Used

- **`interrupt_before=["human_review"]`** — pauses the graph before the review node, waiting for human input
- **`graph.update_state()`** — injects the human's decision (approved/rejected + feedback) into the graph state
- **`graph.invoke(None, config)`** — resumes the graph from the last checkpoint
- **`MemorySaver`** — persists the full graph state across multiple invocations within a session

---

## State
```python
class EmailState(TypedDict):
    topic: str        # what the email is about
    draft: str        # current email draft
    feedback: str     # human feedback on rejection
    approved: bool    # approval status
    iteration: int    # tracks how many drafts were made
```

---

## Project Structure
```
hitl-email-agent/
├── main.py          # core application logic + LangGraph graph
├── .env             # API key (not committed)
├── .gitignore       # excludes .env and cache files
└── README.md        # project documentation
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd hitl-email-agent
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install langgraph langchain langchain-groq python-dotenv
```

### 4. Configure environment
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
python main.py
```

---

## Demo Walkthrough
```
📧 What should the email be about? Apologizing to a client for project delay

⚙️  Drafting email...

────────────────────────────────────────────────────────────
📝  Draft #1:

Subject: Apology for Project Delay

Dear [Client Name],
I sincerely apologize for the delay in delivering your project...
────────────────────────────────────────────────────────────

✅ Approve or ❌ Reject? [a/r]: r
💬 Enter your feedback: Make it more apologetic and mention a discount

⚙️  Redrafting based on your feedback...

────────────────────────────────────────────────────────────
📝  Draft #2:

Subject: Our Sincerest Apologies + Exclusive Discount

Dear [Client Name],
We deeply regret the inconvenience caused by the delay...
────────────────────────────────────────────────────────────

✅ Approve or ❌ Reject? [a/r]: a

✅ Email approved! Final draft saved above.
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Graph orchestration + HITL workflow |
| [LangChain](https://python.langchain.com/) | LLM abstraction layer |
| [Groq](https://console.groq.com/) | Free, fast LLM inference |
| [llama-3.1-8b-instant](https://console.groq.com/docs/models) | Model used for email generation |
| Python 3.9+ | Core language |

---

## Why LangGraph for HITL?

Traditional LLM calls are stateless — once you call the model, it runs to completion.  
LangGraph solves this by treating the AI workflow as a **stateful graph** with checkpointing.  
The `interrupt_before` feature lets the graph pause mid-execution, wait for human input,  
and resume exactly where it left off — making true HITL workflows possible.