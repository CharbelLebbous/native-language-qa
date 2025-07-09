# Import the core pipeline modules: loader, embedder, and QA logic.
from loader import load_documents
from embedder import EmbeddingIndexer
from qa import answer_question
import os

def main():
    """
    Command-line test pipeline for running the Native Language QA prototype.

    Steps:
    1️⃣ Load and chunk all documents in the ./data folder.
    2️⃣ Build the embedding index (in-memory vector store).
    3️⃣ Enter an interactive loop for the user to ask questions.
    4️⃣ Retrieve top similar chunks and generate an answer.
    5️⃣ Print the answer and sources for inspection.
    """

    # Folder containing your documents.
    data_folder = './data'

    print("📚 Loading and chunking documents...")
    docs = load_documents(data_folder)

    print("🧠 Building new embedding index from scratch...")
    indexer = EmbeddingIndexer()

    # Optional: Clear any previously saved index file.
    # (You can ignore this if you're not saving/loading FAISS indexes to disk.)
    if os.path.exists("faiss.index"):
        os.remove("faiss.index")  # Remove old index if exists.

    # Build the vector index using fresh embeddings.
    indexer.build_index(docs)

    # Start an infinite loop for manual testing.
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        # Perform semantic search for the query.
        results = indexer.search(query)

        # Print the top chunks returned for context visibility.
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "Unknown")
            print(f"\nChunk {i+1} (Source: {source}):\n{'-'*40}\n{doc.page_content[:1000]}...")

        # Combine all retrieved chunk texts for answering.
        combined_context = " ".join([doc.page_content for doc in results])

        # Generate an answer from the LLM chain.
        answer = answer_question(query, combined_context)

        # Print the final answer and the sources.
        print(f"\n💬 Answer: {answer}\n")
        print(f"Sources: {[doc.metadata.get('source') for doc in results]}")

# Run the CLI only if this script is called directly.
if __name__ == "__main__":
    main()
