def format_output(state):
    classification = state["classification"]

    sources = []

    for result in state.get("retrieved_results", []):
        chunk = result["chunk"]

        sources.append({
            "source_id": chunk["source_id"],
            "passage": chunk["text"]
        })

    if classification == "requires_escalation":
        requires_human = True
    else:
        requires_human = False

    if classification == "answerable" and state.get("verification_passed"):
        confidence = 0.9
        reason = (
            "The request was answerable using retrieved OrbitDesk "
            "evidence and the generated response passed verification."
        )

    elif classification == "requires_clarification":
        confidence = 0.8
        reason = (
            "The request requires additional information before "
            "it can be answered safely."
        )

    elif classification == "requires_escalation":
        confidence = 0.9
        reason = (
            "The request requires escalation to human support."
        )

    elif classification == "out_of_scope":
        confidence = 0.9
        reason = (
            "The request is outside the scope of OrbitDesk support."
        )

    else:
        classification = "safe_failure"
        confidence = 0.0
        requires_human = True
        reason = (
            state.get("verification_reason")
            or "The response could not be safely verified."
        )

    return {
        "classification": classification,
        "answer": state["answer"],
        "sources": sources,
        "confidence": confidence,
        "requires_human": requires_human,
        "reason": reason
    }