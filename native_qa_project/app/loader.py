# Import standard libraries for file handling and regex cleaning.
import os
import re

# Import LangChain document loaders for different file formats.
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader

# Import LangChain text splitter for chunking long documents into smaller pieces.
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_text(text):
    """
    Cleans and normalizes raw text by:
    - Removing excessive whitespace.
    - Removing invisible unicode characters.
    - Normalizing bullet points or dashes.

    :param text: The raw text to clean.
    :return: Cleaned text as a string.
    """
    # Replace multiple spaces/newlines with single spaces.
    text = re.sub(r'\s+', ' ', text)
    # Remove hidden unicode control characters.
    text = re.sub(r'[\u200b-\u200f]', '', text)
    # Replace various bullet symbols with a simple dash.
    text = re.sub(r'[•–—●]+', '-', text)
    return text.strip()


def load_documents(folder_path):
    """
    Loads and preprocesses all supported documents inside the given folder path.

    Steps:
    - Walk through the folder and subfolders.
    - Detect file extension and choose the right loader (PDF, DOCX, TXT).
    - Clean the text content.
    - Attach useful metadata: file path, file name, and page number.
    - Split the text into smaller chunks for embedding.

    :param folder_path: Path to the folder containing documents.
    :return: A list of chunked Document objects.
    """
    docs = []

    # Initialize a text splitter with small chunk size and overlap.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    # Walk through all files in all subdirectories.
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[-1].lower()

            try:
                # Pick loader based on file extension.
                if ext == ".pdf":
                    loader = PDFPlumberLoader(path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path)
                else:
                    # Skip unsupported file types.
                    continue

                # Load the raw documents using the loader.
                raw_docs = loader.load()

                for doc in raw_docs:
                    # Clean the text content.
                    doc.page_content = clean_text(doc.page_content)

                    # Add useful metadata for traceability.
                    doc.metadata["source"] = path  # Full file path.
                    doc.metadata["file_name"] = file  # File name only.

                    # Try to preserve page number if it’s a PDF.
                    if ext == ".pdf":
                        doc.metadata["page"] = doc.metadata.get("page", "Unknown")
                    else:
                        doc.metadata["page"] = "N/A"

                # Split and add the chunks to the final docs list.
                docs.extend(splitter.split_documents(raw_docs))

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

    return docs
