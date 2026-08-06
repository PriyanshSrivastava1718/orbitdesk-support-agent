from typing import TypedDict, Any
import json

from jsonschema import validate

from langgraph.graph import StateGraph, START, END

from triage import classify_question
from rag import generate_answer
from verifier import verify_answer
from output_formatter import format_output


# ==================================================
# 1. SHARED STATE
# ==================================================

class AgentState(TypedDict):
    question: str
    classification: str
    answer: str
    retrieved_results: list[Any]

    verification_passed: bool
    verification_reason: str
    retry_count: int


# ==================================================
# 2. TRIAGE NODE
# ==================================================

def triage_node(state: AgentState):
    print("NODE: triage")

    classification = classify_question(state["question"])

    return {
        "classification": classification
    }


# ==================================================
# 3. RAG NODE
# ==================================================

def rag_node(state: AgentState):
    print("NODE: rag")

    answer, results = generate_answer(state["question"])

    return {
        "answer": answer,
        "retrieved_results": results
    }


# ==================================================
# 4. VERIFIER NODE
# ==================================================

def verifier_node(state: AgentState):
    print("NODE: verifier")

    passed, reason = verify_answer(
        state["question"],
        state["answer"],
        state["retrieved_results"]
    )

    return {
        "verification_passed": passed,
        "verification_reason": reason
    }


# ==================================================
# 5. REVISION NODE
# ==================================================

def revision_node(state: AgentState):
    print("NODE: revision")

    revised_answer, results = generate_answer(
        state["question"],
        revision_note=state["verification_reason"]
    )

    return {
        "answer": revised_answer,
        "retrieved_results": results,
        "retry_count": state["retry_count"] + 1
    }


# ==================================================
# 6. SAFE FAILURE NODE
# ==================================================

def safe_failure_node(state: AgentState):
    print("NODE: safe_failure")

    return {
        "answer": (
            "I could not produce a response that could be safely "
            "verified against the available OrbitDesk evidence."
        )
    }


# ==================================================
# 7. NON-ANSWERABLE TRIAGE ROUTES
# ==================================================

def clarification_node(state: AgentState):
    print("NODE: clarification")

    return {
        "answer": (
            "I need more information before I can answer this safely."
        )
    }


def escalation_node(state: AgentState):
    print("NODE: escalation")

    return {
        "answer": (
            "This issue requires escalation to human support."
        )
    }


def out_of_scope_node(state: AgentState):
    print("NODE: out_of_scope")

    return {
        "answer": (
            "This request is outside the scope of OrbitDesk support."
        )
    }


# ==================================================
# 8. ROUTING FUNCTIONS
# ==================================================

def route_after_triage(state: AgentState):
    return state["classification"]


def route_after_verification(state: AgentState):

    if state["verification_passed"]:
        return "pass"

    if state["retry_count"] < 1:
        return "retry"

    return "safe_failure"


# ==================================================
# 9. BUILD GRAPH
# ==================================================

builder = StateGraph(AgentState)

builder.add_node("triage", triage_node)
builder.add_node("rag", rag_node)
builder.add_node("verifier", verifier_node)
builder.add_node("revision", revision_node)
builder.add_node("safe_failure", safe_failure_node)

builder.add_node("clarification", clarification_node)
builder.add_node("escalation", escalation_node)
builder.add_node("out_of_scope", out_of_scope_node)


# START -> TRIAGE

builder.add_edge(START, "triage")


# TRIAGE -> CORRECT ROUTE

builder.add_conditional_edges(
    "triage",
    route_after_triage,
    {
        "answerable": "rag",
        "requires_clarification": "clarification",
        "requires_escalation": "escalation",
        "out_of_scope": "out_of_scope"
    }
)


# ANSWERABLE PIPELINE

builder.add_edge("rag", "verifier")


# VERIFIER -> PASS / RETRY / SAFE FAILURE

builder.add_conditional_edges(
    "verifier",
    route_after_verification,
    {
        "pass": END,
        "retry": "revision",
        "safe_failure": "safe_failure"
    }
)


# REVISION -> VERIFIER

builder.add_edge("revision", "verifier")


# SAFE FAILURE -> END

builder.add_edge("safe_failure", END)


# OTHER TRIAGE ROUTES -> END

builder.add_edge("clarification", END)
builder.add_edge("escalation", END)
builder.add_edge("out_of_scope", END)


# COMPILE GRAPH

graph = builder.compile()


# ==================================================
# 10. TEST
# ==================================================

if __name__ == "__main__":

    initial_state = {
        "question": "Can a Viewer create an API credential?",
        "classification": "",
        "answer": "",
        "retrieved_results": [],
        "verification_passed": False,
        "verification_reason": "",
        "retry_count": 0
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 10}
    )

    # Convert internal graph state into assignment output format
    final_output = format_output(result)

    with open("data/output_schema.json", "r", encoding="utf-8") as f:
        output_schema = json.load(f)

    validate(instance=final_output, schema=output_schema)

    print("\nSCHEMA VALIDATION: PASSED")

    print("\nFINAL STATE")

    print("Question:", result["question"])
    print("Classification:", result["classification"])
    print("Answer:", result["answer"])

    print(
        "Verification passed:",
        result["verification_passed"]
    )

    print(
        "Verification reason:",
        result["verification_reason"]
    )

    print(
        "Retry count:",
        result["retry_count"]
    )

    if result["retrieved_results"]:

        print("\nSources:")

        for item in result["retrieved_results"]:
            print(
                "-",
                item["chunk"]["source_id"],
                "| score:",
                round(item["score"], 4)
            )

    print("\nSTRUCTURED OUTPUT")
    print(json.dumps(final_output, indent=2))