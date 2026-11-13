from langchain_ollama import OllamaEmbeddings

from .config import settings

embedder = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)