from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vectorstore/db_faiss'  # Path to the FAISS vectorstore

# Updated prompt template for better response quality
custom_prompt_template = """
You are a helpful assistant with access to a large set of documents.
Your goal is to provide clear, accurate, and detailed responses to the user's questions.

When answering the user's question:
1. Make sure to provide a **detailed explanation** that helps the user understand the context of your answer.
2. If necessary, break down complex ideas into simpler steps.
3. If there is relevant **context** or **information** provided in the documents, use that to support your answer.
4. Always aim to be **clear** and **concise** in your response.
5. If you don’t know the answer, say you don’t know, and do not make up any information.
6. If you are able to use any **source documents** to support your answer, mention them at the end.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else. If there are sources, list them at the end.

Helpful answer:
"""

def set_custom_prompt():
    """
    Returns a prompt template for the QA system.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    """
    Creates a retrieval QA chain that integrates the language model and FAISS vectorstore.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    """
    Loads the Llama model for answering queries.
    """
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",  # Replace with a model you have
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    """
    Initializes the QA bot, combining the model, prompt, and FAISS vectorstore.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    # Allow dangerous deserialization when loading the FAISS index
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def get_answer(query):
    """
    Interacts with the LangChain model and returns an answer to the user's query.
    """
    try:
        qa_result = qa_bot()
        response = qa_result({'query': query})
        return response['result']
    except Exception as e:
        return f"Error processing query: {str(e)}"
