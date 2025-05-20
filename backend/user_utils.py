from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
import joblib
import os
from langchain_groq import ChatGroq


def pull_from_faiss(embedding_model, faiss_index_path = "./faiss_index"):
    if not os.path.exists(faiss_index_path):
        pass
        #Raise FileNotFoundError("FAISS index not found. Please load the data first.")
        #return ["FAISS index not found. Please load the data first."]

    faiss_index = FAISS.load_local(faiss_index_path, embeddings=embedding_model,allow_dangerous_deserialization=True)
    print(f"The data type if faiss index vairiable is {type(faiss_index)}")
    return faiss_index

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
def get_similar_docs(index,query,k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(docs,user_input):
    chain = load_qa_chain(ChatGroq(model = "llama-3.3-70b-versatile"), chain_type="stuff")
    response = chain.invoke({
        "input_documents": docs,
        "question": user_input
    })
    return response["output_text"]

def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pk1')
    result=Fitmodel.predict([query_result])
    return result[0]