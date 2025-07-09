# Import FAISS for efficient vector storage and similarity search,
# and HuggingFaceEmbeddings for generating embeddings from text.
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingIndexer:
    """
    This class handles:
    1. Creating embeddings for documents using a Hugging Face model.
    2. Building a FAISS vector index for efficient similarity search.
    3. Searching for top-k most similar document chunks for a given query.
    """

    def __init__(self):
        # Initialize the embedding model from Hugging Face.
        # Here, we use the 'intfloat/e5-base-v2' model for dense vector embeddings.
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

        # This will store the FAISS index once built.
        self.index = None

    def build_index(self, documents):
        """
        Given a list of chunked documents:
        - Convert them to embeddings.
        - Build a FAISS vector index for similarity search.
        - normalize_L2=True makes FAISS use cosine similarity instead of L2 distance.
        """
        self.index = FAISS.from_documents(
            documents,
            self.embeddings,
            normalize_L2=True  # Using cosine similarity is generally better for embeddings.
        )

    def search(self, query, k=3):
        """
        Perform a similarity search:
        - Embed the input query.
        - Return the top-k most similar document chunks from the index.
        
        :param query: The user question or search string.
        :param k: Number of top results to return.
        :return: List of matched Document chunks.
        """
        return self.index.similarity_search(query, k=k)
