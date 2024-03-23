from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from llama_index.legacy.llms.vertex import Vertex
from llama_index.legacy import (ServiceContext, set_global_service_context)
from enum import Enum
from langchain_google_vertexai import VertexAIEmbeddings

from langchain.chat_models import ChatVertexAI
import os
from dotenv import load_dotenv
load_dotenv()

emb_model_name = "textembedding-gecko-multilingual@001"
gen_model_name  = "text-bison"
service_account_path = os.getenv("SERVICE_ACCOUNT_PATH")

credentials = service_account.Credentials.from_service_account_file(service_account_path)
aiplatform.init(project=credentials.project_id, credentials=credentials)
# embed_model = TextEmbeddingModel.from_pretrained(emb_model_name)
# gen_model = TextGenerationModel.from_pretrained(gen_model_name)
embed_model = VertexAIEmbeddings(model_name='textembedding-gecko-multilingual@latest')

# vertex_ai = Vertex(model="text-bison", project=credentials.project_id, location= "asia-southeast1", credentials=credentials, temperature=0.2)
chat_vertex_ai = ChatVertexAI(model_name="chat-bison-32k", project=credentials.project_id, location= "asia-southeast1", credentials=credentials, temperature=0.2, max_output_tokens= 8192) # max for bison 32k                                 

service_context = ServiceContext.from_defaults(llm=chat_vertex_ai, embed_model=embed_model, chunk_size=1024, chunk_overlap=20)
set_global_service_context(service_context)

class UserIntent(Enum):
    medical_experts=1
    making_appointment=2
    chief_complaint=3
    unknown=4

def get_completion(prompt: str, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 40, max_output_tokens: int = 2048):
        parameters = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_output_tokens': max_output_tokens
        }
        # return gen_model.predict(prompt, **parameters).text
        return chat_vertex_ai.predict(prompt, **parameters)

def get_intent_from_chat(text:str) -> UserIntent:
    prompt = f"Your task is to retrive intention of a given text. You should answer only 'medical experts' when the text is about finding medical expert, 'making appointment' when the text is about making appointment to the medical expert, 'chief complaint' when the text is about symptom. If the text is not related to what previous sentence mentioned, please answer 'unknown'. Text: `{text}`"
    intent = get_completion(prompt).strip()
    if intent in ["medical experts", "making appointment", "chief complaint"]:
        return intent
    return "unknown"


