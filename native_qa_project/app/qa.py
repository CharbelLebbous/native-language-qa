from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain

llm = Ollama(model="phi3:mini", temperature=0.2, num_predict=200) # 1 token ≈ 0.75 words in English, so 200 tokens ≈ 150 words

template = """
You are a helpful AI assistant answering questions based strictly on the provided context.

Your behavior:
- If the question is a greeting (e.g., "hello", "hi", "hey"), respond politely with a greeting.
- If the answer is not found in the context, respond with: "I'm sorry, I couldn't find the answer in the provided documents."
- Do NOT guess or provide information not present in the context.
- Keep answers short, direct, and accurate.

### Examples

Context:
"Welcome to our company documentation."
Question: "hi"
Answer: Hello! How can I assist you today?

Context:
"The device requires a voltage of 12V for operation."
Question: "What is the required voltage?"
Answer: The device requires a voltage of 12V for operation.

Context:
"This document covers safety procedures only."
Question: "What is the warranty period?"
Answer: I'm sorry, I couldn't find the answer in the provided documents.

---

Now use the following context to answer the question:

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

def answer_question(question, context):
    return chain.run({"input_documents": context, "question": question})
