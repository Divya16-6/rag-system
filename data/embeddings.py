from sentence_transformers import SentenceTransformer
from typing import List

def loadModel(modelName: str):
    try:
        model = SentenceTransformer(model_name_or_path=modelName)
        return model
    except Exception as e:
        print(f"Error loading the embedded model {e}")
        raise

def generateEmbeddings(texts: List[str]):
    try:
        model = loadModel("all-MiniLM-L6-v2")
        if model:
            print("Started embedding")
            embeddings = model.encode(texts, show_progress_bar=True)
            print("Embedding completed successfully")
            return embeddings
    except Exception as e:
        print(f"Error generating the embeddings {e}")
        raise
