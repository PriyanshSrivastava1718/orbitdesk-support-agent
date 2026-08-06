import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


_load_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print(f"[verifier.py] Qwen load time: {time.time() - _load_start:.2f}s")


def build_evidence(results):
    evidence_parts = []

    for result in results:
        chunk = result["chunk"]

        evidence_parts.append(
            f"""
Source: {chunk["source_id"]}
Source type: {chunk["source_type"]}
Status: {chunk["status"]}

{chunk["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(evidence_parts)


def verify_answer(question, answer, results):

    # ------------------------------------------
    # DETERMINISTIC CHECK
    # ------------------------------------------

    if not results:
        return False, "No retrieved evidence was provided."

    source_ids = [
        result["chunk"]["source_id"]
        for result in results
    ]

    if not source_ids:
        return False, "No source references were available."

    if not answer.strip():
        return False, "Generated answer was empty."

    # ------------------------------------------
    # MODEL-BASED GROUNDING CHECK
    # ------------------------------------------

    evidence = build_evidence(results)

    messages = [
        {
            "role": "system",
            "content": """
You are a verification component for an OrbitDesk support agent.

Check whether the generated answer is supported by the supplied evidence.

The answer must:
- be grounded in the evidence
- not invent OrbitDesk product behavior
- not invent troubleshooting instructions
- not contradict the supplied evidence

Return exactly one word:

PASS

or

FAIL

Do not explain your decision.
""".strip()
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Evidence:
{evidence}

Generated answer:
{answer}

Verification:
""".strip()
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    _gen_start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False
    )

    print(f"[verifier.py] Verification latency: {time.time() - _gen_start:.2f}s")

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    verdict = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip().upper()

    if verdict == "PASS":
        return True, "Answer is grounded in the retrieved evidence."

    return False, "Answer failed the grounding verification."


if __name__ == "__main__":
    from rag import generate_answer

    question = "Can a Viewer create an API credential?"

    answer, results = generate_answer(question)

    passed, reason = verify_answer(
        question,
        answer,
        results
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print("\nVerification passed:")
    print(passed)

    print("\nReason:")
    print(reason)