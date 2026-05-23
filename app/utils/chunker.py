import re


def split_into_sentences(text: str) -> list[str]:
    return re.split(r'(?<=[.!?])\s+', text)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    sentences = split_into_sentences(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            overlap_chunk: list[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s)

            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
