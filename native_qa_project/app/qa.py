# Import LangChain tools to:
# 1️⃣ Combine documents as context,
# 2️⃣ Create prompts,
# 3️⃣ Run LLM chains with a defined template.
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain

# Initialize the Ollama LLM.
# Uses the local or hosted Ollama server with the specified lightweight model.
# 'temperature' controls randomness; lower is more factual.
# 'num_predict' sets the max token output (~1 token ≈ 0.75 English word).
llm = Ollama(model="phi3:mini", temperature=0.2, num_predict=200)

# Define the system prompt template:
# - Tells the LLM to ONLY answer using context.
# - Handles greetings.
# - Returns a fallback answer if the info is not found.
# - Contains clear examples to guide LLM behavior.
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

# Wrap the template in a LangChain PromptTemplate.
# It dynamically injects context and question at runtime.
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Create an LLM chain that uses the Ollama LLM + custom prompt.
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Wrap it in a StuffDocumentsChain:
# This feeds multiple retrieved document chunks into the prompt.
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

def answer_question(question, context):
    """
    Generates an answer given:
    - 'question': the user input.
    - 'context': list of top document chunks from vector search.

    Runs the LLM chain and returns the model's final answer.
    """
    return chain.run({
        "input_documents": context,
        "question": question
    })
