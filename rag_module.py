import os
import psycopg2
import requests
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()
print("GOOGLE_API_KEY:", os.environ.get("GOOGLE_API_KEY"))

BASE_URL = os.environ.get("BASE_URL")

def load_articles():
    docs = []
    conn = psycopg2.connect(
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT")
    )
    cur = conn.cursor()
    cur.execute('SELECT att_url FROM "Node"')
    rows = cur.fetchall()
    for row in rows:
        pdf_url = row[0]
        if pdf_url.startswith("/"):
            pdf_url = BASE_URL + pdf_url
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            reader = PdfReader(BytesIO(response.content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(text)
        except Exception as e:
            print(f"Failed to process {pdf_url}: {e}")
    cur.close()
    conn.close()
    return docs

def main():
    docs = load_articles()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_texts(chunks, embeddings, persist_directory="chroma_db")
    db.persist()
    print("Ingestion complete.")

def answer_with_rag(question: str):
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        return_source_documents=True,
    )
    result = qa({"query": question})
    return {"answer": result["result"]}

def ingest_pdf_by_url(pdf_url: str):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        reader = PdfReader(BytesIO(response.content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma.from_texts(chunks, embeddings, persist_directory="chroma_db")
        db.add_texts(chunks)
        print(f"Ingestion of {pdf_url} complete.")
        return True
    except Exception as e:
        print(f"Failed to ingest {pdf_url}: {e}")
        return False

if __name__ == "__main__":
    main()
    