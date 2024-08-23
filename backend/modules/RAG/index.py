from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import TiDBVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyD4BaMhXTDgcNJQ5aA3d6Wz6FFqCgleGuQ"

loader=WebBaseLoader("https://en.wikipedia.org/wiki/Mental_health")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
documents

loader1=WebBaseLoader("https://en.wikipedia.org/wiki/Physical_fitness")
docs1=loader1.load()
documents1=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs1)

documents+=documents1
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store1 = TiDBVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    table_name="MentalPhysicalHealthTest",
    connection_string = "mysql+mysqlconnector://2qbX3m6bR7KbhPk.root:EvHYotOPGkRjP18X@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/test",
    distance_strategy="cosine",  # default, another option is "l2"
)

retriever1 = vector_store1.as_retriever(
   score_threshold=0.7
)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = 'AIzaSyD4BaMhXTDgcNJQ5aA3d6Wz6FFqCgleGuQ', temperature=0.7) 

def predict(quest):
    prompt_template = """You are a professional health coach and mental health expert with over 10 years of experience, working in a leading health tech company specializing in providing mental health consultations and diet plans and consultation in order to improves users physical health and overall lifestyle with emphasis on the users being fit both mentally and physically. When answering questions, draw upon your extensive experience while prioritizing the source documents and context provided.

    Instructions for the AI:
    Contextual Learning: Carefully analyze the given source documents and context. Use these sources as your primary reference to formulate detailed, expert-level responses that address the question comprehensively.
    Answer Integration: Combine insights from multiple sections of the provided context when necessary to offer a well-rounded and expert response and do provide short answers with useful tokens and not rubbish tokens like '\n'.
    Source Prioritization: When responding, use as much relevant information from the "response" section of the source documents as possible, maintaining accuracy and detail.
    Fallback Approach: If the context does not provide sufficient information or relevant details, respond with "I don't know."
    Expert Role Assumption: If the context is inadequate and the fallback response isn't triggered, provide advice based on your 10+ years of experience as a professional health coach and mental health expert. Ensure your responses reflect your expertise and professionalism."

        CONTEXT: {context}

        QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever1,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)

    r = chain(quest)

    

    return r