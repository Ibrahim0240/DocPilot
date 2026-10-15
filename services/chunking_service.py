MAX_CHUNKS = 1000


def chunk_document(
    pages: list[dict],
    chunk_size: int = 200,
    overlap: int = 40
) -> list[dict]:

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    chunk_id = 1

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        words = text.split()
        start = 0

        while start < len(words):

            # Prevent excessive chunk creation
            if len(chunks) >= MAX_CHUNKS:
                raise ValueError("Maximum chunk limit exceeded")

            end = min(start + chunk_size, len(words))

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": chunk_text
            })

            chunk_id += 1

            # Stop after processing the final words
            if end == len(words):
                break

            # Calculate where the next chunk begins
            next_start = end - overlap

            # Ensure we're moving forward
            if next_start <= start:
                raise RuntimeError(
                    "Chunking loop is not progressing"
                )

            start = next_start

    return chunks
