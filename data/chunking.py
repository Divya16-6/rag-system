from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    print("Started splitting docs")
    split_docs = text_splitter.split_documents(documents)

    if split_docs:
        print(f"Chunk size {len(split_docs)} for the document size {len(documents)}")

    return split_docs
