import warnings
warnings.filterwarnings("ignore", message=".*TqdmWarning.*")
from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
import sqlite3

from helper import AgentState, Queries, ewriter


class ReviewDecision(BaseModel):
    approved: bool
    feedback: str


class ewriter_with_reviewer(ewriter):
    """Extends the essay writer with a final reviewer agent that approves or
    rejects the draft before ending the workflow."""

    def __init__(self):
        # Initialise the parent (builds the base graph nodes/prompts/model)
        super().__init__()

        self.REVIEWER_PROMPT = (
            "You are a senior editor performing a final quality review of a 3-paragraph essay. "
            "Evaluate the essay on the following criteria:\n"
            "  1. Clarity – is the writing clear and easy to follow?\n"
            "  2. Depth – does it go beyond surface-level observations?\n"
            "  3. Structure – does it follow the planned outline with a proper introduction, body, and conclusion?\n"
            "  4. Accuracy – are any factual claims supported by the research content provided?\n\n"
            "If the essay meets a high standard across all criteria, set approved=True and summarise why.\n"
            "If it falls short, set approved=False and give specific, actionable feedback so the writer can improve it."
        )

        # Rebuild the graph to include the reviewer node
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.add_node("reviewer", self.reviewer_node)

        builder.set_entry_point("planner")
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: "reviewer", "reflect": "reflect"},
        )
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        builder.add_conditional_edges(
            "reviewer",
            self.reviewer_decision,
            {"approved": END, "revise": "reflect"},
        )

        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=[
                "planner",
                "generate",
                "reflect",
                "research_plan",
                "research_critique",
                "reviewer",
            ],
        )

    def reviewer_node(self, state: AgentState):
        """Final reviewer – approves the draft or asks for another revision."""
        content_summary = "\n\n".join(state["content"] or [])
        user_message = HumanMessage(
            content=(
                f"Essay topic: {state['task']}\n\n"
                f"Research content used:\n{content_summary}\n\n"
                f"Final draft:\n{state['draft']}"
            )
        )
        decision: ReviewDecision = self.model.with_structured_output(ReviewDecision).invoke(
            [SystemMessage(content=self.REVIEWER_PROMPT), user_message]
        )
        return {
            "critique": decision.feedback,
            "lnode": "reviewer",
            "count": 1,
        }

    def reviewer_decision(self, state: AgentState) -> str:
        """Route to END if reviewer approved, otherwise send back for revision."""
        # The reviewer stored its decision in `critique`; we check for approval
        # by re-evaluating – simpler: re-invoke structured output on the stored feedback.
        feedback = state.get("critique", "")
        # A lightweight heuristic: if the reviewer's feedback begins with a known
        # approval phrase we stored, treat it as approved.  For a robust solution
        # we persist the boolean in state; here we re-check via the model.
        decision: ReviewDecision = self.model.with_structured_output(ReviewDecision).invoke(
            [
                SystemMessage(
                    content=(
                        "You previously reviewed the essay. Based on the feedback below, "
                        "was the essay approved (approved=True) or not (approved=False)? "
                        "Reproduce your decision faithfully."
                    )
                ),
                HumanMessage(content=feedback),
            ]
        )
        return "approved" if decision.approved else "revise"


if __name__ == "__main__":
    import time

    agent = ewriter_with_reviewer()
    thread = {"configurable": {"thread_id": "1"}}
    topic = "The impact of artificial intelligence on modern healthcare"
    initial_state = {
        "task": topic,
        "max_revisions": 2,
        "revision_number": 0,
        "lnode": "",
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "queries": [],
        "count": 0,
    }

    print(f"Running essay writer with reviewer for topic: '{topic}'\n")
    for event in agent.graph.stream(initial_state, thread):
        for node, value in event.items():
            lnode = value.get("lnode", node)
            print(f"[{lnode}] completed")
            if lnode == "generate":
                draft = value.get("draft", "")
                if draft:
                    print(f"  Draft (rev {value.get('revision_number', '?')}):\n  {draft[:200]}...\n")
            elif lnode == "reviewer":
                print(f"  Reviewer feedback: {value.get('critique', '')[:200]}\n")
