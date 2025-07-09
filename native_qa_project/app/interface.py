from loader import load_documents
from embedder import EmbeddingIndexer
from qa import answer_question
import os

def main():
    data_folder = './data'

    print("📚 Loading and chunking documents...")
    docs = load_documents(data_folder)

    print("🧠 Building new embedding index from scratch...")
    indexer = EmbeddingIndexer()

    # ❗ Clear any saved index file (if you had implemented save/load logic)
    # If you're not using saved FAISS files, you can skip this
    # But if yes, consider deleting the saved file here
    if os.path.exists("faiss.index"):
        os.remove("faiss.index")  # optional: clear old saved index

    indexer.build_index(docs)

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        results = indexer.search(query)
        
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "Unknown")
            print(f"\nChunk {i+1} (Source: {source}):\n{'-'*40}\n{doc.page_content[:1000]}...")

        combined_context = " ".join([doc.page_content for doc in results])
        answer = answer_question(query, combined_context)

        print(f"\n💬 Answer: {answer}\n")
        print(f"\n💬 Answer: {answer}\nSources: {[doc.metadata.get('source') for doc in results]}")


if __name__ == "__main__":
    main()
