# backend/summarizer.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from transformers import pipeline
from pdf2image import convert_from_path
from PIL import Image
from docx import Document as DocxDocument
import io
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def load_document(file_path):
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use .pdf or .docx")
    return loader.load()

def summarize_document(file_path):
    print("Loading document...")
    documents = load_document(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    from transformers import pipeline

    summarizer_pipeline = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        token=HUGGINGFACE_TOKEN
    )

    llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
    prompt = PromptTemplate.from_template(
        "Summarize this text clearly and concisely:\n\n{text}"
    )

    summarizer_chain = (
        RunnableMap({"text": lambda x: x.page_content})
        | prompt
        | llm
        | StrOutputParser()
    )

    summaries = []
    for chunk in chunks:
        try:
            out = summarizer_chain.invoke(chunk)
            summaries.append(out)
        except Exception as e:
            summaries.append(chunk.page_content[:500] + "...")

    return "\n\n".join(summaries)
