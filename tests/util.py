import re
import subprocess
import glob

import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

collection_name="chat_with_docs"

#hosts a local vector database
client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def clone_github_repo(repo_url):
    try:
        print('Cloning the repo ...')
        result = subprocess.run(["git", "clone", repo_url], check = True, text = True, capture_output = True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return None
    
def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def parse_docs_by_file_types(ext, language, input_dir_path):
    try:
        files = glob.glob(f"{input_dir_path}/**/*{ext}", recursive = True)

        if len(files) > 0:
            loader = SimpleDirectoryReader(
                input_dir = input_dir_path, required_exts=[ext], recursive = True
            )
            
            docs = loader.load_data()

            parser = (
                MarkdownNodeParser()
                if ext == '.md'
                else CodeSplitter.from_defaults(language=language)
            )
            
            return parser.get_nodes_from_documents(docs)
        else:
            return []
    except Exception as e:
        print(f'Exception {e} occurred while parsing docs into nodes of file type {ext}')
        return []
    
def create_index(nodes):
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes, 
        storage_context = storage_context
    )
    return index


