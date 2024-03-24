from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader
from llama_index.legacy.embeddings import LangchainEmbedding
from llama_index.legacy.text_splitter import TokenTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from llama_index.legacy.llms.vertex import Vertex
from llama_index.legacy.node_parser import SimpleNodeParser, HierarchicalNodeParser
from llama_index.legacy import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    download_loader,
    load_index_from_storage,
    VectorStoreIndex)
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.prompts import (
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)

from llama_index.legacy.postprocessor import NERPIINodePostprocessor, SentenceEmbeddingOptimizer
from llama_index.legacy import ServiceContext
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.schema import NodeWithScore, TextNode

from pathlib import Path
from tqdm.notebook import tqdm
from google.oauth2 import service_account

from llama_index.legacy import set_global_service_context
import re
import uuid
import os
from pathlib import Path
from pprint import pprint
import pandas as pd
import csv
from typing import List, Tuple, Dict
import time
import json

from llama_index.legacy.vector_stores.chroma import ChromaVectorStore
from llama_index.legacy import StorageContext
from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from llama_index.legacy.llms.langchain import LangChainLLM
from langchain_google_vertexai import ChatVertexAI
from llama_index.legacy import Response
from llama_index.legacy.response_synthesizers import Refine
from llama_index.legacy.evaluation import SemanticSimilarityEvaluator
from llama_index.legacy.evaluation import RelevancyEvaluator
from llama_index.legacy.embeddings import SimilarityMode

import nest_asyncio
nest_asyncio.apply()

from llama_index.legacy.postprocessor import LongContextReorder
from llama_index.legacy.retrievers import AutoMergingRetriever
from llama_index.legacy.query_engine import TransformQueryEngine
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.indices.query.query_transform.base import (
    HyDEQueryTransform,StepDecomposeQueryTransform
)
from llama_index.legacy.node_parser import get_leaf_nodes
from llama_index.legacy import Document

import log_appointment as logapp
from get_intent import chat_vertex_ai


# credentials = service_account.Credentials.from_service_account_file("credentials/vertex-test-417403-ce72ad032af7.json")
# vertex_ai = Vertex(model="text-bison", project=credentials.project_id, location= "asia-southeast1", credentials=credentials, temperature=0.2)
# langchain_chat_vertex_ai = LangChainLLM(ChatVertexAI(model_name="chat-bison@002", project=credentials.project_id, location= "asia-southeast1", credentials=credentials, temperature=0.2))
# selected_model = langchain_chat_vertex_ai
# embed_model = LangchainEmbedding(VertexAIEmbeddings(model_name='textembedding-gecko-multilingual@latest'))

# service_context = ServiceContext.from_defaults(llm=vertex_ai, embed_model=embed_model, chunk_size=1024, chunk_overlap=20)
# set_global_service_context(service_context)


# select index
clinic_doctor_index_dir = os.getenv("CLINIC_INDEX_DIR")
loaded_storage_context = StorageContext.from_defaults(persist_dir=clinic_doctor_index_dir) # load the existing index
clinic_doctor_index = load_index_from_storage(loaded_storage_context)

# create retreiver
automerging_as_retriever = clinic_doctor_index.as_retriever(similarity_top_k=10)
automerging_retriever = AutoMergingRetriever(
    automerging_as_retriever, 
    clinic_doctor_index.storage_context, 
    verbose=True
)

# create query engine
reorder = LongContextReorder()
hyde = HyDEQueryTransform(llm=chat_vertex_ai, include_original=True)
retriever_query_engine = RetrieverQueryEngine.from_args(automerging_retriever,
                                              node_postprocessors=[reorder],
                                              )
transform_query_engine = TransformQueryEngine(retriever_query_engine, query_transform=hyde)


pii_processor = NERPIINodePostprocessor()

def filter_pii_fn(**kwargs):
    # run optimizer
    query_bundle = QueryBundle(query_str=kwargs["query_str"])

    new_nodes = pii_processor.postprocess_nodes(
        [NodeWithScore(node=TextNode(text=kwargs["context_str"]))],
        query_bundle=query_bundle,
    )
    new_node = new_nodes[0]
    return new_node.get_content()


qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "If you need to state yes or no, state it in Thai only\n"
    "If the question is a yes/no question, answer either yes or no in Thai before giving your reasoning for the answer.\n"
    "The reasoning should contain the relevant information and be concise if possible. Don't reiterate the same information twice.\n"
    "If the question asks about the date and time for an apppointment, answer in the following format only.\n"
    "{day_time} {full_name} ({expertise})\n"
    "for example: วันจันทร์ เวลา 09:00 - 13:00 น. doctor: รศ. พญ.กติกา นวพันธุ์ expertise: เวชศาสตร์ฟื้นฟู \n" 
    "Also, the format of the answer should be as similar to the format in the context information as possible. This includes numbering order and indentation. \n"
    "The answer should not include the name of the document where the information is gotten from.\n"
    "However, if the context information does not contain the answer for the query, answer เอกสารไม่มีระบุไว้\n"
    "Query: {query_str}\n"
    "Answer: "
    "1. {day_time} {full_name} ({expertise})\n"
    "2. {day_time} {full_name} ({expertise})\n"
    "..."
)

qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": filter_pii_fn}
)

prompts_dict = transform_query_engine.get_prompts()
print(list(prompts_dict.keys()))

transform_query_engine.update_prompts(
    {"query_engine:response_synthesizers": qa_prompt_tmpl}
)


# main function to get output
def get_response(query):
    print("query", query)
    response = transform_query_engine.query(query)
    print("retrieval", response)
    # retrieved_nodes = automerging_retriever.retrieve(query)
    # context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    return response #, context_str