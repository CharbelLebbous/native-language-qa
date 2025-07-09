from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingIndexer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
        self.index = None

    def build_index(self, documents):
        # ✅ Use cosine similarity by normalizing vectors
        self.index = FAISS.from_documents(
            documents,
            self.embeddings,
            normalize_L2=True  # 🔥 Enables cosine similarity instead of L2 distance
        )

    def search(self, query, k=3):
        return self.index.similarity_search(query, k=k)