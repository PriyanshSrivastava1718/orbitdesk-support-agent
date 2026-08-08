import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

ALLOWED_LABELS = {
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
}

_load_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    torch_dtype="auto",
    device_map="auto"
)

print(
    f"[triage.py] Qwen 3B load time: "
    f"{time.time() - _load_start:.2f}s"
)


def classify_question(question):

    system_prompt = """
You are a strict classification system for OrbitDesk customer support.

Classify the user's REQUEST into exactly ONE of these four labels:

answerable
requires_clarification
requires_escalation
out_of_scope

Definitions:

answerable:
The request is a clear OrbitDesk product or support question.
There is enough information to begin answering it using the
OrbitDesk knowledge base or resolved support cases.
A troubleshooting question can be answerable even when the user
has not provided every diagnostic detail.

requires_clarification:
The request is about OrbitDesk but is too vague to determine
what feature, problem, or situation the user means.

requires_escalation:
The user explicitly states that documented troubleshooting
has already been completed and the problem still persists,
or the request clearly requires human support.

A first report of a failure is NOT automatically escalation.
If the user asks what to check or what to do next, it can be
answerable.

out_of_scope:
The request is unrelated to OrbitDesk support or asks for
something outside the support agent's capabilities, such as
issuing a refund or providing legal advice.

Examples:

Request: Can a Viewer create an API credential?
Label: answerable

Request: Our data sync is not working.
Label: requires_clarification

Request: We already checked the dashboard, connections and
destination. Two export runs failed with render_failed.
Label: requires_escalation

Request: Issue a refund and give me legal advice.
Label: out_of_scope

Request: Our daily dashboard exports stopped appearing at the
expected time after an Admin changed the workspace timezone
yesterday. The schedule still looks active. What should we check,
and can the missed export be recovered?
Label: answerable

Request: How does the workspace timezone affect scheduled exports,
and where can scheduled exports be delivered?
Label: answerable

Return ONLY the label.
Do not explain your decision.
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
        "How does the workspace timezone affect scheduled exports, and where can scheduled exports be delivered?",
        "I'm a Viewer and need an API credential for a reporting job. Can I create one?",
        "My dashboard connection stopped syncing data.",
        "We verified the schedule, destination, timezone and connections. The export failed again with the same error.",
        "Can you refund my payment because the report failed?",
        "After changing the workspace timezone, my recurring export runs at the wrong time. What should I check?",
        "Are scheduled exports affected by the workspace timezone?"
    ]

    for question in test_questions:
        result = classify_question(question)

        print("\nQuestion:", question)
        print("Triage:", result)
        print("-" * 60)