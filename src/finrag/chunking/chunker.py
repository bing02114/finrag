
class TextChunker:

    @staticmethod
    def fixed_size(
        text: str,
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> list[str]:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be between 0 and chunk_size")

        chunks = []
        step = chunk_size - overlap

        for start in range(0, len(text), step):
            chunk = text[start:start + chunk_size]

            if chunk.strip():
                chunks.append(chunk)

            if start + chunk_size >= len(text):
                break

        return chunks

    @staticmethod
    def recursive(
        text: str,
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> list[str]:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= overlap < chunk_size:
            raise ValueError("Invalid overlap")

        separators = ["\n\n", "\n", "。", "；", "，", ""]

        def split_text(text, separator_index=0):

            if len(text) <= chunk_size:
                return [text] if text.strip() else []

            separator = separators[separator_index]

            if separator == "":
                return [
                    text[i:i + chunk_size]
                    for i in range(0, len(text), chunk_size)
                ]

            parts = text.split(separator)
            chunks = []
            buffer = ""

            for part in parts:
                candidate = (
                    buffer + separator + part
                    if buffer else part
                )

                if len(candidate) <= chunk_size:
                    buffer = candidate
                else:
                    if buffer.strip():
                        chunks.append(buffer)

                    if len(part) > chunk_size:
                        chunks.extend(
                            split_text(
                                part,
                                separator_index + 1,
                            )
                        )
                        buffer = ""
                    else:
                        buffer = part

            if buffer.strip():
                chunks.append(buffer)

            return chunks

        chunks = split_text(text)

        # Add overlap using the end of previous chunk
        if overlap > 0:
            result = []

            for i, chunk in enumerate(chunks):
                if i == 0:
                    result.append(chunk)
                else:
                    prefix = chunks[i - 1][-overlap:]
                    result.append(prefix + chunk)

            return result

        return chunks
