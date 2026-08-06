import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


messages = [
    {
        "role": "system",
        "content": "You are a concise customer support assistant."
    },
    {
        "role": "user",
        "content": "Explain in one sentence what an API credential is."
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
    max_new_tokens=80,
    do_sample=False
)


generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)


print("\nModel response:")
print(response)