from fastapi import FastAPI, File, UploadFile, HTTPException
from data.pdf_loader import pdfReader
from data.chunking import split_documents
from data.embeddings import generateEmbeddings
from data.vector_store import addDataToTheStore, createOrGetCollection
from data.retriever import ragRetriever, generateResponse
from data.model.retriever_model import QueryModel, ResponseModel

app = FastAPI()

@app.post("/process/doc")
def processDoc(file: UploadFile = File(...)):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")
        file.file.seek(0)
        if not file.file.read(1):
            raise HTTPException(status_code=400, detail="File uploaded is empty")
        file.file.seek(0)

        file_bytes = file.file.read()
        # Document Reader
        docs = pdfReader(file_bytes, file.filename)

        # Splitting the document (Chunking)
        split_docs = split_documents(documents=docs)

        # Embedding the docs
        texts = [doc.page_content for doc in split_docs]
        embeddings = generateEmbeddings(texts)

        addDataToTheStore(split_docs, embeddings, collectionName="sample_python_store")
        data = createOrGetCollection("sample_python_store")
        return {
        "pages": len(docs),
        "chunks": len(split_docs),
        "sample_chunk": split_docs[0].page_content[:300] if split_docs else "",
        "collection count": data.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/retrieve")
def retrieve(query: str, top_k: int = 5):
    try:
        data = ragRetriever(query, top_k)
        return {
            "data": data["documents"][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/response")
def generateResponseWithContext(query_request: QueryModel):
    try:
        response = generateResponse(query=query_request.query)
        return ResponseModel(content=response.content, questions=response.questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))