from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from utils.file_proceessor_for_indexing import FileProcessorForIndexing

from dotenv import load_dotenv
load_dotenv()


def get_vector_store():
    client = QdrantClient(url="http://localhost:6333",port=6333)
    if not client.collection_exists("sanath_projects_latest"):
        print("Creating Collection")
        index_helper = FileProcessorForIndexing()
        file_texts = index_helper.process_all_files()
        documents = index_helper.chunk_all_files(file_texts)
        qdrant = QdrantVectorStore.from_documents(
            documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name="sanath_projects_latest",
            url="http://localhost:6333",
            port=6333
        )
    else:
        print("Collection exists")
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name="sanath_projects_latest",
            url="http://localhost:6333",
            port=6333
        )

    return qdrant


    

qdrant_store = get_vector_store()