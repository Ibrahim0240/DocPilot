from ollama import Client

client = Client(
    host="http://localhost:11434",
    timeout=120
)

EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE = 8


def embed_chunks(chunks: list[dict]) -> list[dict]:

    for start in range(0, len(chunks), list[dict]):
        batch = chunks[start:start+BATCH_SIZE]

        texts = ["Search Document"+chunk['text']
                 for chunk in batch
                 ]
        response = client.embed(
            model=EMBEDDING_MODEL,
            input=texts,
            truncate=False
        )

        embeddings = response["embeddings"]

        if len(embeddings) != len(batch):
            raise ValueError("Embedding count does not match chunk count")

        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding

    return chunks
