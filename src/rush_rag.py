from pathlib import Path
import textwrap
import logging
import chromadb 
from pypdf import PdfReader
from Ollam import Client

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR/ "data" / "pdfs"

