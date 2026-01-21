from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os


def pdfReader(file_bytes: bytes, fileName: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    print("Processing the file")
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata['source_file'] = fileName
        doc.metadata['file_type'] = "pdf"
        doc.metadata['title'] = "Python"
        doc.metadata['creator'] = "Divya S"
        
    os.remove(tmp_path)
    print("Loader completed")
    
    return documents
