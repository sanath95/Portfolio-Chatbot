from glob import glob
from json import load
from pathlib import Path
from re import compile, MULTILINE
from pymupdf import Document as PdfDocument

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, url="http://localhost:6333", port=6333):
        self.url = url
        self.port = port

    def get_vector_store(self):
        client = QdrantClient(url=self.url,port=self.port)
        
        if not client.collection_exists("sanath_projects_latest"):
            qdrant = self.__create_vector_store()
        else:
            qdrant = self.__get_vector_store()

        return qdrant
    
    def __create_vector_store(self):
        index_helper = FileProcessorForIndexing()
        file_texts = index_helper.process_all_files()
        documents = index_helper.chunk_all_files(file_texts)
        qdrant = QdrantVectorStore.from_documents(
            documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name="sanath_projects_latest",
            url=self.url,
            port=self.port
        )

        return qdrant
    
    def __get_vector_store(self):
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name="sanath_projects_latest",
            url=self.url,
            port=self.port
        )
        return qdrant

class FileProcessorForIndexing:
    def __init__(self):
        self.INPUT_FOLDER = "./data/*"

        self.WHITESPACE_PATTERN = compile(r"\s{2,}")
        self.IMAGE_PATTERN = compile(r'!\[[^\]]*\]\([^)\s]+(?:\s+"[^"]*")?\)')
        self.MULTIPLE_NEWLINE_PATTERN = compile(r'\n+')
        self.MD_HEADING_PATTERN = compile(r"(?=^#{1,6}\s+)", MULTILINE)

        PDF_CHUNK_SIZE = 2090
        MD_CHUNK_SIZE = 1172
        PDF_OVERLAP_SIZE = 200
        MD_OVERLAP_SIZE = 0
        self.splitter_config = {
            '.pdf': (PDF_CHUNK_SIZE, PDF_OVERLAP_SIZE),
            '.md': (MD_CHUNK_SIZE, MD_OVERLAP_SIZE)
        }

        with open("./configs/data_config.json") as f:
            self.DATA_CONFIG = load(f)

    def process_all_files(self):
        text_documents = {}
        files = glob(self.INPUT_FOLDER)
        for file in files:
            text_documents[file] = self.__process_file(file)

        return text_documents

    def __process_file(self, file):
        filepath = Path(file)
        
        if filepath.suffix == '.pdf':
            cleaned_text_sections = self.__process_pdf(file)
        
        elif filepath.suffix == '.md':
            cleaned_text_sections = self.__process_markdown(file)

        return cleaned_text_sections

    def __process_pdf(self, filepath):
        doc = PdfDocument(filepath)
        text_sections = [page.get_text("text") for page in doc]
        return self.__clean_text_sections(text_sections)
    
    def __process_markdown(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        text_sections = self.MD_HEADING_PATTERN.split(text)
        return self.__clean_text_sections(text_sections)
    
    def __clean_text_sections(self, text_sections):
        cleaned_sections = (self.__clean_text(text) for text in text_sections if text.strip())
        return cleaned_sections
    
    def __clean_text(self, text):
        text = ''.join(c for c in text if c.isprintable())
        
        text = self.WHITESPACE_PATTERN.sub('\n', text)
        text = self.IMAGE_PATTERN.sub('', text)
        text = self.MULTIPLE_NEWLINE_PATTERN.sub('\n', text)
        text = text.replace("---", "")
        
        return text
    
    def chunk_all_files(self, file_texts):
        documents = []
        for doc, contents in file_texts.items():
            ext = Path(doc).suffix
            
            if ext not in self.splitter_config:
                continue
                
            chunk_size, overlap_size = self.splitter_config[ext]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=overlap_size
            )
            metadata = self.DATA_CONFIG[Path(doc).name]

            for page_content in contents:
                texts = text_splitter.split_text(page_content)
                for page_text in texts:
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata=metadata
                        )
                    )

        return documents