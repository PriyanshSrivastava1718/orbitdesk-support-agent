from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentence = "Can a Viewer create an API credential?"

embedding = model.encode(sentence)

print("Sentence:")
print(sentence)

print("\nEmbedding shape:")
print(embedding.shape)

print("\nFirst 10 values:")
print(embedding[:10])