def chunk_pages(pages, chunk_size: int = 500):
    chunks = []

    for page_data in pages:
        text = page_data["text"]
        page_number = page_data["page"]

        sentences = text.split(".")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                chunks.append({
                    "text": current_chunk,
                    "page": page_number
                })
                current_chunk = sentence + "."

        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "page": page_number
            })

    return chunks