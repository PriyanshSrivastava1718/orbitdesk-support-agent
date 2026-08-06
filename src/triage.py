import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ALLOWED_LABELS = {
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
}


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


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
A clear OrbitDesk product or support question that can proceed
to the knowledge-grounded answer pipeline.

requires_clarification:
The request concerns OrbitDesk, but it is too vague or is missing
important information needed to proceed.

requires_escalation:
The request states that troubleshooting has already been performed
and failed repeatedly, or clearly describes a situation requiring
human support.

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

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip().lower()

    if prediction in ALLOWED_LABELS:
        return prediction

    return "requires_clarification"


if __name__ == "__main__":

    test_questions = [
        "Can a Viewer create an API credential?",
        "Our data sync is not working.",
        "We already checked the dashboard, connections and destination. Two export runs failed with render_failed.",
        "Issue a refund and give me legal advice."
    ]

    for question in test_questions:
        result = classify_question(question)

        print("\nQuestion:", question)
        print("Triage:", result)
        print("-" * 60)