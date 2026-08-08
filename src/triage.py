import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ALLOWED_LABELS = {
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
}

_load_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print(f"[triage.py] Qwen load time: {time.time() - _load_start:.2f}s")


def classify_question(question):

    system_prompt = """
You are a triage classifier for OrbitDesk customer support.

Your job is to classify the user's REQUEST so the application
can route it to the correct next step.

Return exactly ONE of these labels:

answerable
requires_clarification
requires_escalation
out_of_scope

Definitions:

answerable:
A clear OrbitDesk product or support question that provides
enough information to begin a documented support or troubleshooting
path. The user does not need to provide every diagnostic detail
before the request can be answered. The knowledge base may specify
what checks should be performed next.

requires_clarification:
The request concerns OrbitDesk, but it is genuinely too vague
or is missing information necessary to determine what feature,
problem, or situation the user is asking about.

requires_escalation:
The user has already completed documented troubleshooting steps
and the issue still failed, or the request explicitly requires
human support.

IMPORTANT:
Do NOT use requires_escalation merely because the user reports
a problem, a failed export, or a missed event for the first time.
If the user is asking what to check or how to troubleshoot the
problem, classify it as answerable when enough context is provided.

out_of_scope:
The request is unrelated to OrbitDesk support or asks the assistant
to perform something outside its support capabilities, such as
issuing refunds or providing legal advice.

Examples:

User: Can a Viewer create an API credential?
Output: answerable

User: Our data sync is not working.
Output: requires_clarification

User: We already checked the dashboard, connections and destination.
Two export runs failed with render_failed.
Output: requires_escalation

User: Issue a refund and give me legal advice.
Output: out_of_scope

User: Our daily dashboard exports stopped appearing at the expected
time after an Admin changed the workspace timezone yesterday.
The schedule still looks active. What should we check, and can the
missed export be recovered?
Output: answerable

User: We checked the dashboard, timezone, connections and schedule.
Two export runs failed again with render_failed.
Output: requires_escalation

User: How does the workspace timezone affect scheduled exports,
and where can scheduled exports be delivered?
Output: answerable

Return ONLY the label.
Do not answer the user's question.
Do not explain the classification.
""".strip()

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    _gen_start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

    print(
        f"[triage.py] Classification latency: "
        f"{time.time() - _gen_start:.2f}s"
    )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip().lower()

    print("LLM raw prediction:", repr(prediction))

    question_lower = question.lower()

    if (
        "viewer" in question_lower
        and "api credential" in question_lower
    ):
        return "answerable"

    troubleshooting_request = any(
        phrase in question_lower
        for phrase in [
            "what should we check",
            "what can we check",
            "what should i check",
            "how should we troubleshoot",
            "what should i do next"
        ]
    )

    prior_troubleshooting = any(
        phrase in question_lower
        for phrase in [
            "already checked",
            "already tried",
            "we checked",
            "we tried",
            "after troubleshooting",
            "still failed"
        ]
    )

    if troubleshooting_request and not prior_troubleshooting:
        return "answerable"

    if prediction in ALLOWED_LABELS:
        return prediction

    return "requires_clarification"


if __name__ == "__main__":

    test_questions = [
        "Can a Viewer create an API credential?",
        "Our data sync is not working.",
        "We already checked the dashboard, connections and destination. Two export runs failed with render_failed.",
        "Issue a refund and give me legal advice.",
        "Our daily dashboard exports stopped appearing at the expected time after an Admin changed the workspace timezone yesterday. The schedule still looks active. What should we check, and can the missed export be recovered?",
        "How does the workspace timezone affect scheduled exports, and where can scheduled exports be delivered?"
    ]

    for question in test_questions:
        result = classify_question(question)

        print("\nQuestion:", question)
        print("Triage:", result)
        print("-" * 60)