"""Document processing utilities for converting files to vector store format."""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.config import ProcessorConfig


class FileProcessor:
    """Processes PDF and Markdown files into chunked LangChain documents.
    
    Attributes:
        config: Processor configuration.
        files: List of file paths matching the input glob.
        metadata_config: Per-file metadata from configuration.
        splitter: Markdown header-based text splitter.
        pdf_converter: PDF to Markdown converter.
    """

    def __init__(self, config: ProcessorConfig) -> None:
        """Initialize file processor.
        
        Args:
            config: Processor configuration.
        """
        self.config = config
        self.files = glob(str(config.input_glob))
        self.metadata_config = self._load_metadata_config()
        self.splitter = self._create_markdown_splitter()
        self.pdf_converter = self._create_pdf_converter()

    def build_documents(self) -> list[Document]:
        """Build LangChain documents from all input files.
        
        Returns:
            List of processed and chunked documents with enriched metadata.
        """
        documents: list[Document] = []
        for file_path in self.files:
            documents.extend(self._process_file(file_path))
        return documents

    def _process_file(self, file_path: str) -> list[Document]:
        """Process a single file into chunked documents with metadata.
        
        Args:
            file_path: Path to the file to process.
            
        Returns:
            List of processed Document objects with enriched content and metadata.
        """
        path = Path(file_path)
        markdown_text = self._to_markdown(path)
        docs = self.splitter.split_text(markdown_text)
        return self._enrich_documents(docs, path.name)

    def _enrich_documents(
        self, docs: list[Document], file_name: str
    ) -> list[Document]:
        """Enrich documents with metadata in both content and metadata fields.
        
        Args:
            docs: List of documents to enrich.
            file_name: Name of the source file for metadata lookup.
            
        Returns:
            Enriched documents with prepended metadata in page_content and
            updated metadata dictionary.
        """
        metadata = self.metadata_config.get(file_name, {})

        # Build prefix from file name, tools, and skills
        prefix_parts = [
            file_name,
            *metadata.get("tools_used", []),
            *metadata.get("skills", []),
        ]
        prefix = " ".join(prefix_parts)

        # Enrich each document
        for doc in docs:
            doc.page_content = f"{prefix}\n{doc.page_content}"
            doc.metadata.update(metadata)

        return docs

    def _to_markdown(self, path: Path) -> str:
        """Convert a file to Markdown text.
        
        Args:
            path: Path to the input file.
            
        Returns:
            Markdown representation of the file.
            
        Raises:
            ValueError: If the file format is unsupported.
        """
        if path.suffix == ".pdf":
            result = self.pdf_converter.convert(str(path))
            return result.document.export_to_markdown()
        elif path.suffix == ".md":
            return path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_metadata_config(self) -> dict[str, dict[str, Any]]:
        """Load per-file metadata configuration.
        
        Returns:
            Mapping of filename to metadata dictionary.
        """
        with open(self.config.config_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _create_markdown_splitter() -> MarkdownHeaderTextSplitter:
        """Create a Markdown header-based text splitter.
        
        Returns:
            Configured MarkdownHeaderTextSplitter.
        """
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            strip_headers=False,
        )

    def _create_pdf_converter(self) -> DocumentConverter:
        """Create a PDF-to-Markdown document converter.
        
        Returns:
            Configured DocumentConverter for PDF input.
        """
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        do_ocr=True,
                        ocr_options=EasyOcrOptions(lang=["en"]),
                        do_table_structure=False,
                        accelerator_options=AcceleratorOptions(
                            num_threads=self.config.num_threads,
                            device=AcceleratorDevice.CUDA,
                        ),
                    )
                )
            }
        )