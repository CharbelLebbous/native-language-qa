import os
import re
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text):
    """Remove unwanted characters and normalize whitespace."""
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces/newlines
    text = re.sub(r'[\u200b-\u200f]', '', text)  # Strip invisible unicode
    text = re.sub(r'[•–—●]+', '-', text)  # Normalize bullets/dashes
    return text.strip()

def load_documents(folder_path):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[-1].lower()

            try:
                if ext == ".pdf":
                    loader = PDFPlumberLoader(path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path)
                else:
                    continue

                raw_docs = loader.load()

                for doc in raw_docs:
                    # Clean text
                    doc.page_content = clean_text(doc.page_content)

                    # ✅ Set clear metadata
                    doc.metadata["source"] = path  # full file path
                    doc.metadata["file_name"] = file

                    # ✅ Assign page number for PDFs
                    if ext == ".pdf":
                        doc.metadata["page"] = doc.metadata.get("page", "Unknown")
                    else:
                        doc.metadata["page"] = "N/A"

                docs.extend(splitter.split_documents(raw_docs))

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

    return docs
