import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from retrieval import retrieve


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


def build_context(results):
    context_parts = []

    for result in results:
        chunk = result["chunk"]

        source = chunk["source_id"]
        source_type = chunk["source_type"]
        status = chunk["status"]
        text = chunk["text"]

        context_part = f"""
Source: {source}
Source type: {source_type}
Status: {status}

{text}
""".strip()

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question):
    results = retrieve(question, top_k=3)

    context = build_context(results)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an OrbitDesk customer-support assistant. "
                "Answer using only the provided evidence. "
                "Do not invent product behavior or troubleshooting steps. "
                "Treat current knowledge-base documentation as authoritative. "
                "Historical resolved cases may be used only as supporting evidence. "
                "If the evidence is insufficient, say that you do not have enough "
                "information to answer safely."
            )
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Evidence:
{context}

Answer the question concisely.
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

    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer, results


if __name__ == "__main__":
    question = "Can a Viewer create an API credential?"

    answer, results = generate_answer(question)

    print("\nQuestion:")
    print(question)

    print("\nSources used:")
    for result in results:
        print(
            "-",
            result["chunk"]["source_id"],
            "|",
            result["chunk"]["source_type"],
            "| score:",
            round(result["score"], 4)
        )

    print("\nAnswer:")
    print(answer)