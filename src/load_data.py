import json
from pathlib import Path

kb_folder = Path("knowledge_base")

print("Looking for KB at:", kb_folder.resolve())
print("KB folder exists:", kb_folder.exists())

kb_files = sorted(kb_folder.glob("*.md"))

print(f"Found {len(kb_files)} knowledge-base files.\n")
documents = []
for file_path in kb_files:

    content = file_path.read_text(encoding="utf-8")
    document = {
        "filename": file_path.name,
        "text": content
    }
    documents.append(document)
    print(f"Loaded: {file_path.name}")
    print(f"Characters: {len(content)}")
    print("-" * 40)
cases_path = Path("data/resolved_cases.json")

with cases_path.open("r", encoding="utf-8") as file:
    resolved_cases = json.load(file)

cases = resolved_cases["cases"]

for case in cases:
    print(
        case["case_id"],
        "->",
        case["status"]
    )

print(f"Loaded {len(cases)} resolved cases.")
print(cases[0])

print(f"\nStored {len(documents)} documents in memory.")