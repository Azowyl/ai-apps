from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

class DocumentHandler:
    def __init__(self) -> None:
        self.loader = DirectoryLoader('./docs', glob="**/*.pdf", loader_cls=PyPDFLoader)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def upload_documents(self):
        raw_docs = self.loader.load()
        documents = self.text_splitter.split_documents(raw_docs)
        print(f"Going to add {len(documents)} chunks to Pinecone")

        PineconeVectorStore.from_documents(documents=documents, embedding=self.embeddings, index_name=os.getenv('INDEX_NAME'))
        print("Loading to vectorstore done")

    def retrieve_relevant_documents(self, query):
        document_vectorstore = PineconeVectorStore(index_name=os.getenv('INDEX_NAME'), embedding=self.embeddings)
        retriever = document_vectorstore.as_retriever()
        return retriever.get_relevant_documents(query)

