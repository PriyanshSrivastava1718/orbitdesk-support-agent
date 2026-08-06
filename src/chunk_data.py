from pathlib import Path
import re

import json

def extract_metadata(text):
    metadata = {}

    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) >= 3:
            metadata_block = parts[1]

            for line in metadata_block.strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

    return metadata


def remove_metadata(text):
    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) >= 3:
            return parts[2].strip()

    return text.strip()


def chunk_markdown(file_path):
    raw_text = file_path.read_text(encoding="utf-8")

    metadata = extract_metadata(raw_text)
    clean_text = remove_metadata(raw_text)

    sections = re.split(r"\n## ", clean_text)

    chunks = []

    for index, section in enumerate(sections):
        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()

        if index == 0:
            section_title = metadata.get("title", file_path.stem)
        else:
            section_title = lines[0].strip()
            section = "\n".join(lines[1:]).strip()

        if not section:
            continue

        chunk = {
            "chunk_id": f"{metadata.get('document_id', file_path.stem)}-{index}",
            "source_id": metadata.get("document_id", file_path.stem),
            "source_type": "knowledge_base",
            "filename": file_path.name,
            "section": section_title,
            "status": metadata.get("status", "unknown"),
            "text": section
        }

        chunks.append(chunk)

    return chunks


def load_kb_chunks():
    kb_folder = Path("knowledge_base")
    all_chunks = []

    for file_path in sorted(kb_folder.glob("*.md")):
        file_chunks = chunk_markdown(file_path)
        all_chunks.extend(file_chunks)

    return all_chunks

def load_case_chunks():
    cases_path = Path("data/resolved_cases.json")

    with cases_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    cases = data["cases"]

    case_chunks = []

    for case in cases:

        symptoms = "\n".join(
            f"- {item}" for item in case.get("symptoms", [])
        )

        resolution = "\n".join(
            f"- {item}" for item in case.get("resolution", [])
        )

        text = f"""
Title: {case.get("title", "")}

Symptoms:
{symptoms}

Resolution:
{resolution}

Important limit:
{case.get("important_limit", "")}
""".strip()

        chunk = {
            "chunk_id": case["case_id"],
            "source_id": case["case_id"],
            "source_type": "resolved_case",
            "status": case.get("status", "unknown"),
            "product_version": case.get("product_version"),
            "source_documents": case.get("source_documents", []),
            "text": text
        }

        case_chunks.append(chunk)

    return case_chunks

if __name__ == "__main__":
    kb_chunks = load_kb_chunks()
    case_chunks = load_case_chunks()

    print(f"Created {len(kb_chunks)} KB chunks.")
    print(f"Created {len(case_chunks)} historical case chunks.")

    print("\nHistorical cases:\n")

    for chunk in case_chunks:
        print("=" * 60)
        print("Case:", chunk["source_id"])
        print("Status:", chunk["status"])
        print("Supporting KB:", chunk["source_documents"])
        print()
        print(chunk["text"][:300])
        print()