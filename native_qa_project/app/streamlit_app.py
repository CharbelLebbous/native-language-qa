import streamlit as st
import os
import pickle
from loader import load_documents
from embedder import EmbeddingIndexer
from qa import answer_question
from langchain_community.vectorstores import FAISS

st.title("📚 Native Language QA on Documents")

# Exit App button
col1, col2, col3 = st.columns([4, 1, 1])
with col3:
    if st.button("❌ Exit App"):
        st.warning("App terminated.")
        st.stop()


# Folder path input
folder_path = st.text_input("📁 Enter path to document folder:", "./data")

# Load Folder button
if st.button("📂 Load Folder"):
    if (
        "indexer" not in st.session_state
        or folder_path != st.session_state.get("loaded_path", "")
    ):
        st.session_state.loaded_path = folder_path
        st.session_state.indexer = None
        st.session_state.docs = None

        with st.spinner("🔄 Loading & building index from folder..."):
            st.session_state.docs = load_documents(folder_path)
            if not st.session_state.docs:
                st.error("❌ No valid documents found in the selected folder.")
            else:
                st.session_state.indexer = EmbeddingIndexer()
                st.session_state.indexer.build_index(st.session_state.docs)

# Ask a question
if "indexer" in st.session_state and st.session_state.indexer:
    query = st.text_input("🔎 Ask a question:")

    if query:
        results = st.session_state.indexer.search(query)

        # Build detailed source info for each result
        sources = []
        for doc in results:
            file_name = doc.metadata.get("file_name", "Unknown File")
            page = doc.metadata.get("page", "N/A")
            source_path = doc.metadata.get("source", "Unknown Path")
            source_str = f"📄 **{file_name}** — Page: {page}  \n📁 Path: `{source_path}`"
            sources.append(source_str)

        # Deduplicate while keeping order
        unique_sources = list(dict.fromkeys(sources))

        st.write("🔍 **Top Contexts Retrieved:**")
        for i, doc in enumerate(results):
            st.text_area(f"Chunk {i+1} (From: {sources[i]})", doc.page_content[:500], height=150)

        with st.spinner("🤖 Generating answer..."):
            answer = answer_question(query, results)

        st.markdown(f"**💬 Answer:** {answer}")
        st.markdown("**📄 Sources Used:**")
        for src in unique_sources:
            st.markdown(src)

else:
    st.info("👆 Select a folder and press **Load Folder** to start.")
