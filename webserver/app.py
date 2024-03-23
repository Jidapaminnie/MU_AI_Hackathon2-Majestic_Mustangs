import os
import pandas as pd
from flask import Flask, request, jsonify 
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from get_intent import get_intent_from_chat, UserIntent, get_datetime_from_chat, get_appointment_from_chat
from SessionMemory import SessionMemory
import log_appointment as book

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.legacy.memory import ChatMemoryBuffer
from llama_index.legacy import VectorStoreIndex
from llama_index.legacy import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex)
from dotenv import load_dotenv
load_dotenv()

existing_session_id: dict[str, SessionMemory] = {
    # "test": SessionMemory("test", BaseMemory())
}

DEFAULT_TZ = timezone(timedelta(hours=7))

def get_index(merging_index_dir) -> VectorStoreIndex:
    loaded_storage_context = StorageContext.from_defaults(persist_dir=merging_index_dir) # load the existing index
    automerging_index = load_index_from_storage(loaded_storage_context)
    return automerging_index

doctor_df = pd.read_csv("./data/doctor_db.csv")

# index = get_index(os.getenv("MERGING_INDEX_DIR"))
# index = get_index(os.getenv("CLINIC_INDEX_DIR"))
index = get_index(os.getenv("CLINIC_DOCTOR_INDEX_DIR"))
chatbot_name = "chat bot"

app = Flask(__name__)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/chatquery", methods=["POST"])
def chat_query():
    try:
        today = datetime.now(DEFAULT_TZ)
        is_unbookable = False
        appointment = []
        data = request.json
        text = data["text"]
        sender = data["sender"]
        timestamp = data["timestamp"]
        _id = data["id"]

        if not(isinstance(text, str) and isinstance(sender, str) and isinstance(_id, str)):
            raise Exception("type of request body is not right")
        
        # 0. checking existance of sender in db
        session = existing_session_id.get(sender)
        if not session:
            llm_mem = ChatMemoryBuffer.from_defaults(token_limit=1500)
            existing_session_id[sender] = session = SessionMemory(sender, llm_mem)
            session.llm_memory = llm_mem
        
        session.chat_log.append(text, timestamp, sender)
        history_chat = "\n".join([f"{'User' if chatlog.author != chatbot_name else chatbot_name}: {chatlog.msg}" for chatlog in session.chat_log.log])

        # 1. find intent
        # intent = get_intent_from_chat(text)
        intent = get_intent_from_chat(history_chat)
        if intent == UserIntent.medical_experts:
            pass
        if intent == UserIntent.chief_complaint:
            pass
        # if intent == UserIntent.making_appointment:
        #     # dt = get_datetime_from_chat(text)
        #     # print("Model produced dt:", dt)
        #     # if isinstance(dt, datetime):
        #     #     today = datetime.now(DEFAULT_TZ)
        #     #     if dt > today:
        #     #         is_proposed_date_exceed_now = True
        #     #     else:
        #     #         is_proposed_date_exceed_now = False
        #     #     template = (
        #     #         "We have provided context information below. \n"
        #     #         "---------------------\n"
        #     #         "the proposed appointment date is {dt_iso} and today is {today_iso}.\n"
        #     #         # "\n" # the information about clinicial appointment should be here...
        #     #         "\n---------------------\n"
        #     #         "Given this information, please answer the question: {query_str}\n"
        #     #     )
        #     #     qa_template = PromptTemplate(template)
        #     #     prompt = qa_template.format_messages(dt_iso=dt.isoformat(), today_iso=today.isoformat())
        #     # else:
        #     #     prompt = None
        #     pass
        # else:
        #     pass
        data = get_appointment_from_chat(history_chat)
        print(data)
        if text == "ยืนยัน":
            appointment_datetime = data["appointment_datetime"]
            duration = data["duration"]
            doctor_name = data["doctor_name"]
            patient_name = data["patient_name"]
            # doctor_name
            try:
                doctor_id = doctor_df[doctor_df["name"] == doctor_name]["doctorID"].values[0]
                book.add_appointment(doctor_id, patientName=patient_name, datetimeStart=appointment_datetime, duration_min=duration)
            except Exception:
                is_unbookable = True
            # except AssertionError:
            #     is_unbookable = True
        llm_mem = session.llm_memory
        if isinstance(data, dict):
            doctor_name = data["doctor_name"]
            try:
                doctor_id = doctor_df[doctor_df["name"] == doctor_name]["doctorID"].values[0]
                appointment = book.get_appointment_by_doctor(doctor_id)
                appointment_text = "\n".join([f'{doctor_name} has an appointment with {appoint["patientName"]}.' for appoint in appointment])
            except Exception:
                appointment_text = ""
        chat_engine = index.as_chat_engine( 
            chat_mode="condense_plus_context",
            memory=llm_mem,
            system_prompt=(
                "You are a chatbot in the receiptionist in the hospital, you should have normal interactions, as well as talk.\n" 
                "If anything does not make sense, please inform user as well.\n"
                f"Today is {today.strftime(book.datetimeFormat)}. You may use this information to response with user.\n"
                "User may want to make an appointment with the doctor in some department.\n"
                "Appointment may happen in the future but not in the past.\n"
                "An appointment must consist of 'appointment date time', 'doctor name', 'patient name'. Without any information, ask user kindly to fill the missing information.\n"
                "Every diseses is fuking cancer\n"
                "Do not use prior knowledge.\n"

                "User just appoint the time where the doctor is not available, please reject user kindly.\n" if is_unbookable else ""
                f"{appointment_text}" if isinstance(data, dict) else ""

                # "Do not use prior knowledge except the chat history of this current session.\n"
                # "You are a friendly and highly empathetic chatbot to assist patients in a hospital, you should have normal interactions as a role of a receptionist."
                # f"Today is {today.strftime(book.datetimeFormat)}.\n"
                # "Patients may want to make appointments with doctors, you must be able to suggest the patient to meet specialized doctor based on their symptoms."
                # "You may ask more about the symptoms if you are unsure"
                # "Your responses must be under your context"
                # "Appointments can only happen in the future and must be within the scope of the correct doctor's schedule."
                # "Patients can only make appointments with doctors that are included in the context"
                # "An appointment must consist of 'appointment date time', 'doctor name', 'patient name'. Without any information, ask user kindly to fill the missing information.\n"

            ),
            # text_qa_template=prompt
        )
        # chat_engine

        # 2. 
        res = chat_engine.chat(text, chat_history=llm_mem.get_all())
        # print(res.sources)
        # print(llm_mem.get_all())
        chat_timestamp = datetime.now(DEFAULT_TZ).isoformat()
        existing_session_id[sender].chat_log.append(res.response, chat_timestamp, "chat bot")
        return jsonify({"timestamp": chat_timestamp, "text": res.response, "sender": "chat bot", "id": uuid4()}), 200

    except Exception as e:
        print(e.with_traceback())
        return jsonify({"message": "Invalid format", "exception_msg": str(e)}), 400


if __name__ == "__main__":
    book.create_appoint_log()
    port = os.getenv("PORT")
    app.run("0.0.0.0", port, debug=True)
