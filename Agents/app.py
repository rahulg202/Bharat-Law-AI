import os
from typing import List, Dict, Tuple
import warnings
import langdetect  
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")

os.environ["GOOGLE_API_KEY"] = "Key"  

VECTOR_STORE_BASE_PATH = "./VectorStore"
VECTOR_STORES = {
    "Copyright": os.path.join(VECTOR_STORE_BASE_PATH, "CV/"),
    "GI": os.path.join(VECTOR_STORE_BASE_PATH, "GV/"),
    "Design": os.path.join(VECTOR_STORE_BASE_PATH, "DV/"),
    "Patent": os.path.join(VECTOR_STORE_BASE_PATH, "PV/"),
    "Trademark": os.path.join(VECTOR_STORE_BASE_PATH, "TV/")
}
app = FastAPI(title="Multilingual IP Law Expert API")
templates = Jinja2Templates(directory="templates")

def detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except:
        return "en"


def translate_text(text: str, target_language: str) -> str:
    translator_llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1
    )

    if target_language == "en":
        translation_prompt = f"""
        Translate the following text to English. Preserve the meaning and technical terms.

        Text: {text}

        Translation:
        """
    else:
        translation_prompt = f"""
        Translate the following text to {target_language}. Preserve the meaning and technical terms.

        Text: {text}

        Translation:
        """

    response = translator_llm.invoke(translation_prompt)
    return response.strip()


def classify_ip_domain(query: str) -> str:
    classifier_llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1
    )

    classification_prompt = f"""
    Classify the following query into ONE of these Intellectual Property Law domains:
    - GI (Geographical Indications)
    - Trademark
    - Patent
    - Design
    - Copyright

    Return ONLY the category name without explanation.

    Query: {query}
    """

    response = classifier_llm.invoke(classification_prompt)
    domain = response.strip()

    if "copyright" in domain.lower():
        return "Copyright"
    elif "gi" in domain.lower() or "geographical" in domain.lower():
        return "GI"
    elif "design" in domain.lower():
        return "Design"
    elif "patent" in domain.lower():
        return "Patent"
    elif "trademark" in domain.lower():
        return "Trademark"
    else:
        print(f"Classification failed, defaulting to Copyright. LLM response: {domain}")
        return "Copyright"


def load_vector_store(domain: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    vector_store_path = VECTOR_STORES.get(domain)

    if not vector_store_path:
        raise ValueError(f"Invalid domain: {domain}")

    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store not found at {vector_store_path}")

    try:
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded from {vector_store_path}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise


def setup_rag_chain(vector_store):
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2
    )

    template = """
    You are an expert in Intellectual Property Law. Answer the following question based on the provided context.

    Context: {context}

    Question: {question}

    Please provide a detailed, authoritative answer using the context information. Include relevant legal principles, 
    case references, and practical implications when applicable. 

    If the information in the context is insufficient to fully answer the question, acknowledge this but still provide 
    the best answer possible based on what is available. Do not make up information.

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # Modified to return source documents
    )

    return qa_chain


async def answer_query(query: str) -> Dict:
    try:
        source_language = detect_language(query)
        print(f"Detected language: {source_language}")

        if source_language != "en":
            english_query = translate_text(query, "en")
            print(f"Translated query: {english_query}")
        else:
            english_query = query

        domain = classify_ip_domain(english_query)
        print(f"Classified query as {domain} domain")

        vector_store = load_vector_store(domain)
        qa_chain = setup_rag_chain(vector_store)

        response = qa_chain.invoke({"query": english_query})
        english_answer = response["result"]
        source_documents = response.get("source_documents", [])

        sources = []
        for doc in source_documents:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])

        if source_language != "en":
            translated_answer = translate_text(english_answer, source_language)
            result = translated_answer
        else:
            result = english_answer

        return {"result": result, "domain": domain, "sources": sources}

    except Exception as e:
        error_message = f"An error occurred while processing your query: {str(e)}"

        if source_language != "en":
            translated_error = translate_text(error_message, source_language)
            return {"result": translated_error, "domain": "Error", "sources": []}
        return {"result": error_message, "domain": "Error", "sources": []}


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query")
async def process_query(query: str = Form(...)):
    """Process the user's query and return the response"""
    response = await answer_query(query)
    return JSONResponse(content=response)

if not os.path.exists("templates"):
    os.makedirs("templates")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
