from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFChunker:
    """
    Handles PDF loading and chunking.

    Parameters:
    - k: chunk size (controlled by UI)
    - chunk_overlap: fixed to 200
    """

    CHUNK_OVERLAP = 200

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and return one Document per page.
        """
        loader = PyPDFLoader(pdf_path)

        documents = loader.load()
        if not documents:
            raise RuntimeError(f"No content extracted from PDF: {pdf_path}")

        return documents

    def _get_splitter(self, k: int) -> RecursiveCharacterTextSplitter:
        """
        Create a text splitter using chunk size `k`
        and fixed chunk overlap.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=k,
            chunk_overlap=self.CHUNK_OVERLAP,
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def extract(
        self,
        pdf_path: str,
        k: int = 1000,
    ) -> List[Document]:
        """
        Load and chunk a PDF file.

        Args:
            pdf_path: Local path to PDF
            k: Chunk size

        Returns:
            List[Document]: Chunked LangChain Documents
        """
        documents = self._load_pdf(pdf_path)
        splitter = self._get_splitter(k)

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise RuntimeError(f"Chunking produced no output for PDF: {pdf_path}")

        return chunks
