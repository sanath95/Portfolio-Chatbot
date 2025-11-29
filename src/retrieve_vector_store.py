from utils.vector_store import VectorStore

from dotenv import load_dotenv
load_dotenv()

    

vs = VectorStore()

qdrant_store = vs.get_vector_store()