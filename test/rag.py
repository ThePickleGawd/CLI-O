#saves cache of the models to the weight folder 
import os  
os.environ["HF_HOME"] = os.path.expanduser("~/weights")
os.environ["TORCH_HOME"] = os.path.expanduser("~/weights")


from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate

from util import *
from IPython.display import Markdown, display



#initializes local LLM
#sets it LLM as default LLM and Hugging Face model as default embed_model
llm = Ollama(
    model ="llama3", 
    base_url="http://localhost:11434",
    request_timeout = 120.0
)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-base-en-v1.5")

def setup_query_engine(github_url):
    owner, repo = parse_github_url(github_url)

    if validate_owner_repo(owner, repo):

        input_dir_path = os.path.join(os.getcwd(), repo)

        if os.path.exists(input_dir_path):
            pass
        else:
            clone_github_repo(github_url)
        
        try:
            file_types = {
                ".md": "markdown",
                ".py": "python",
                ".ipynb": "python",
                ".js": "javascript",
                ".ts": "typescript"
            }

            nodes = []
            for ext, language in file_types.items():
                nodes += parse_docs_by_file_types(ext, language, input_dir_path)

            try:
                index = create_index(nodes)
            except:
                print("FAILED!")
                index = VectorStoreIndex(nodes=nodes, show_progress=True)
            
            query_engine = index.as_query_engine(similarity_top_k=4)

            qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )

            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )

            if nodes:
                print("Data Loaded Successfully!")
                print("Ready to Chat!")
            else:
                print("No Data Found, check if repo is not empty!")
            
            return query_engine
        
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Invalid github repo, try again")
        return None




github_url = "https://github.com/Lightning-AI/LitServe"

query_engine = setup_query_engine(github_url=github_url)

#response = query_engine.query('Can you provide a step by step guide to finetuning an llm using lit-gpt')

retriever = query_engine.retriever
retriever.similarity_top_k = 4

try:
    nodes = retriever.retrieve('Can you provide a step by step guide to finetuning an llm using lit-gpt')
    context = "\n\n".join([node.node.get_content() for node in nodes])
    print(context)
except Exception as e:
    print(f"[RAG Context Retrieval Failed] {e}")






