import chromadb
import os
import uuid

def createOrGetCollection(collectionName: str):
    client = createClient()

    collection = client.get_or_create_collection(
        name=collectionName,
        metadata={"description": "PDF document embeddings for RAG"}
    )
    print(f"Created the collection using the collection name: {collectionName}")
    return collection


def addDataToTheStore(documents, embeddings, collectionName):
    if len(documents) != len(embeddings):
        raise ValueError("Given length of the documents and embedding are not equal")
    
    ids = []
    metadatas = []
    documents_text = []
    embeddings_list = []

    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        doc_id = str(uuid.uuid4())
        ids.append(doc_id)

        metadata = dict(doc.metadata)
        metadata['doc_index'] = i
        metadata['content_length'] = len(doc.page_content)
        metadatas.append(metadata)

        documents_text.append(doc.page_content)

        embeddings_list.append(embedding.tolist())
    
    collection = createOrGetCollection(collectionName)
    collection.add(
        ids=ids,
        metadatas=metadatas,
        embeddings=embeddings_list,
        documents=documents_text
    )
    print("Successfully added the data's into the collection of the vector store")

def createClient():
    directory = "../data/store"
    os.makedirs(directory, exist_ok=True)
    print("Creating the chromadb client")
    client = chromadb.PersistentClient(directory)
    return client