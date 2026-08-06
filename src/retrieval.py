from sentence_transformers import SentenceTransformer
from chunk_data import load_kb_chunks, load_case_chunks


# Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Load both sources
kb_chunks = load_kb_chunks()
case_chunks = load_case_chunks()

# Combine them into one searchable collection
chunks = kb_chunks + case_chunks


# Extract only the text that needs to be embedded
chunk_texts = [chunk["text"] for chunk in chunks]


# Convert all chunks into normalized embedding vectors
chunk_embeddings = model.encode(
    chunk_texts,
    normalize_embeddings=True
)


def retrieve(question, top_k=3):

    # Convert the user's question into a vector
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # Compare the question with every chunk
    similarities = chunk_embeddings @ question_embedding

    # Superseded historical cases must not be used
    # as current support guidance
    for index, chunk in enumerate(chunks):
        if (
            chunk["source_type"] == "resolved_case"
            and chunk["status"] == "superseded"
        ):
            similarities[index] = -1.0

    # Sort scores from highest to lowest
    # and keep only the Top-K results
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        result = {
            "score": float(similarities[index]),
            "chunk": chunks[index]
        }

        results.append(result)

    return results


if __name__ == "__main__":

    question = "OrbitDesk banana thing isn't working."

    results = retrieve(question)

    print("\nQuestion:")
    print(question)

    print("\nTop retrieved chunks:\n")

    for result in results:

        chunk = result["chunk"]

        print("=" * 60)
        print("Similarity:", round(result["score"], 4))
        print("Source:", chunk["source_id"])
        print("Source type:", chunk["source_type"])
        print("Status:", chunk["status"])
        print("Section:", chunk.get("section", "Historical case"))
        print()
        print(chunk["text"])
        print()