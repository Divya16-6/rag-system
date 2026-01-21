from data.vector_store import createOrGetCollection
from data.embeddings import generateEmbeddings
from data.llm_loader import generateLLMResponse

def ragRetriever(query: str, top_k: int):
    query_embedding = generateEmbeddings([query])[0]
    try:
        client = createOrGetCollection("sample_python_store")
        results = client.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results
    except Exception as e:
        raise


def generateResponse(query: str):
    try:
        results = ragRetriever(query, 5)
        documents = results["documents"][0]
        context = "\n\n".join(documents) 
        response = generateLLMResponse(context, query)
        return response
    except Exception as e:
        raise