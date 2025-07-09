# ✅ Import core modules:
# - streamlit: web UI framework
# - os, pickle: standard Python libs (pickle is unused here)
# - loader, embedder, qa: your custom pipeline modules
# - FAISS: vector store class (imported but not directly used here)
import streamlit as st
import os
import pickle  # Note: not used in current code
from loader import load_documents
from embedder import EmbeddingIndexer
from qa import answer_question
from langchain_community.vectorstores import FAISS

# 🚀 Set the page title
st.title("📚 Native Language QA on Documents")

# ✅ Create a 3-column layout to place the exit button on the right
col1, col2, col3 = st.columns([4, 1, 1])
with col3:
    if st.button("❌ Exit App"):
        st.warning("App terminated.")
        st.stop()  # Stops the Streamlit script immediately

# 📂 Text input for the folder path with a default suggestion
folder_path = st.text_input("📁 Enter path to document folder:", "./data")

# 📂 Load Folder button:
# 1️⃣ Load all files from folder if not loaded yet.
# 2️⃣ If path changed, clear old index and reload.
if st.button("📂 Load Folder"):
    if (
        "indexer" not in st.session_state
        or folder_path != st.session_state.get("loaded_path", "")
    ):
        # Remember the folder path to detect changes later
        st.session_state.loaded_path = folder_path
        st.session_state.indexer = None
        st.session_state.docs = None

        with st.spinner("🔄 Loading & building index from folder..."):
            # Call your custom loader to read and chunk all valid docs
            st.session_state.docs = load_documents(folder_path)

            if not st.session_state.docs:
                st.error("❌ No valid documents found in the selected folder.")
            else:
                # Initialize the embedder and build the FAISS index
                st.session_state.indexer = EmbeddingIndexer()
                st.session_state.indexer.build_index(st.session_state.docs)

# ✅ If index is ready, show question input and answer output
if "indexer" in st.session_state and st.session_state.indexer:
    query = st.text_input("🔎 Ask a question:")

    if query:
        # 🔍 Search the vector store for top-k similar chunks
        results = st.session_state.indexer.search(query)

        # 🗂️ Build metadata for source display (file name, page, path)
        sources = []
        for doc in results:
            file_name = doc.metadata.get("file_name", "Unknown File")
            page = doc.metadata.get("page", "N/A")
            source_path = doc.metadata.get("source", "Unknown Path")
            source_str = f"📄 **{file_name}** — Page: {page}  \n📁 Path: `{source_path}`"
            sources.append(source_str)

        # ✅ Remove duplicates but keep order
        unique_sources = list(dict.fromkeys(sources))

        # 📝 Show the top retrieved contexts (text chunks)
        st.write("🔍 **Top Contexts Retrieved:**")
        for i, doc in enumerate(results):
            st.text_area(
                f"Chunk {i+1} (From: {sources[i]})",
                doc.page_content[:500],
                height=150
            )

        # 🧠 Run LLM to generate the final answer
        with st.spinner("🤖 Generating answer..."):
            answer = answer_question(query, results)

        # ✅ Show final answer + sources
        st.markdown(f"**💬 Answer:** {answer}")
        st.markdown("**📄 Sources Used:**")
        for src in unique_sources:
            st.markdown(src)

else:
    # 📌 Reminder for the user to load a folder first
    st.info("👆 Select a folder and press **Load Folder** to start.")
