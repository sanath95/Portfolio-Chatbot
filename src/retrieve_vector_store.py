from utils.file_proceessor_for_indexing import VectorStore

from dotenv import load_dotenv
load_dotenv()

    

vs = VectorStore()

qdrant_store = vs.get_vector_store()