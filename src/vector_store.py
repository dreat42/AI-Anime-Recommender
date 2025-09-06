from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_dir: str = "chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def build_and_save_vectorstore(self):
        # Load CSV and convert to LangChain documents
        loader = CSVLoader(
            file_path=self.csv_path,
            encoding='utf-8',
            metadata_columns=[]  # You can include relevant metadata if needed
        )

        data = loader.load()

        # Split text into manageable chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_documents(data)

        # Build vector store (Chroma persists automatically now)
        db = Chroma.from_documents(
            documents=texts,
            embedding=self.embedding,
            persist_directory=self.persist_dir
        )

        # No need for db.persist() — deprecated in Chroma 0.4.x+
        # db.persist()

    def load_vector_store(self):
        # Load existing vector store from disk
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )
