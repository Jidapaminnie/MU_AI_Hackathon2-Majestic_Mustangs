from datetime import datetime, timezone, timedelta
import dateparser
import json
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
DEFAULT_TZ = timezone(timedelta(hours=7))

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
    prompt = f"Your task is to retrive user intention of a given text. The text will contain history of user and chat bot. You should answer only 'medical experts' when the text is about finding medical expert, 'making appointment' when the text is about making appointment to the medical expert, 'chief complaint' when the text is about symptoms and opening hours. The most informative text is probably user's latest message. If the text is not related to what previous sentence mentioned, please answer 'unknown'.\nChatlog:\n{text}"
    intent = get_completion(prompt).strip()
    if intent == "medical experts":
        return UserIntent.medical_experts
    if intent == "making appointment":
        return UserIntent.making_appointment
    if intent == "chief complaint":
        return UserIntent.chief_complaint
    return UserIntent.unknown

def get_datetime_from_chat(text:str):
    prompt = f"Your task is to retrive datetime of a given text. You should answer only a datetime string in ISO format, eg. '03 March 2024 at 8:08 PM' should be turn into '2024-03-23T20:08:00'. If the text is not related to what previous sentence mentioned, please answer 'unknown'. Text: `{text}`"
    dt = get_completion(prompt).strip()
    try:
        dt = dt.fromisoformat()
        return dt
    except Exception:
        # print(f"cant parse dt({dt})")
        return "unknown"

def get_appointment_from_chat(chat_history:str):
    prompt = ("Your task is to summarize a medical appointment of a given chat history between a user and chat bot.\n"
              "You should be able to make a summary in JSON from consist of the following topic 'appointment datetime', 'duration', 'doctor name', 'patient name'\n"
              "The appointment datetime must be in ISO format, eg. '03 มีนาคม 2024 2 ทุ่ม 8 นาที' should be turn into '2024-03-23T20:08:00'.\n"
              "If the user does not specify the duration of the appointment, make it default to 1 hours. And write it in term of minutes, eg. '1 ชม. ครึ่ง' should be '90'.\n"
              "If anything you do not know about any of the topic, just put `idk`.\n"
              "Chat history will be written in this template '`Name`: `Text`' alternate between User and chat bot\n"
              "Chat history will be written below\n"
              "======================"
              f"{chat_history}"
              "======================")
    print(chat_history)
    data = get_completion(prompt).strip() # hopefully json
    data = "\n".join(data.split("\n")[1: -1])
    # raw res is surrounded with ```
    print("raw data from prompt", data)
    try:
        data = json.loads(data)
        data["appointment_datetime"] = dateparser.parse(data["appointment_datetime"])
        return data
    except Exception:
        return "fukng stupid"

def get_date_from_chat(text:str):
    prompt = ("Your task is to find a date time related keyword.\n"
              "For example for input \n"
              "The appointment datetime must be in ISO format, eg. '03 March 2024 at 8:08 PM' should be turn into '2024-03-23T20:08:00'.\n"
              "If the user does not specify the duration of the appointment, make it default to 1 hours. And write it in term of minutes, eg. '1 ชม. ครึ่ง' should be '90'.\n"
              "Chat history will be written in this template '`Name`: `Text`' alternate between User and chat bot\n"
              "Chat history will be written below\n"
              "======================"
              f"{text}"
              "======================")
    # print(text)
    data = get_completion(prompt).strip() # hopefully json
    data = "\n".join(data.split("\n")[1: -1])
    # raw res is surrounded with ```
    # print(data)
    try:
        data = json.loads(data)
        return data
    except Exception:
        return "fukng stupid"

