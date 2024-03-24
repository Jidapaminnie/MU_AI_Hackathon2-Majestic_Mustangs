import os
import pandas as pd
from flask import Flask, request, jsonify 
from datetime import datetime, timezone, timedelta
from dateparser import parse
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
from retrieval_model import get_response, qa_prompt_tmpl_str, filter_pii_fn, transform_query_engine

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
index_cli = get_index(os.getenv("CLINIC_INDEX_DIR"))
index_cli_doc = get_index(os.getenv("CLINIC_DOCTOR_INDEX_DIR"))
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
        is_week = False
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
            session = existing_session_id[sender] = SessionMemory(sender, llm_mem)
            session.llm_memory = llm_mem
        
        session.chat_log.append(text, timestamp, sender)
        history_chat = "\n".join([f"{'User' if chatlog.author != chatbot_name else chatbot_name}: {chatlog.msg}" for chatlog in session.chat_log.log])
        llm_mem = session.llm_memory

        # 1. find intent
        intent = get_intent_from_chat(history_chat)
        print("intent", intent)
        if (intent == UserIntent.medical_experts) or (intent == UserIntent.chief_complaint):
            # res = get_response(text)

            # qa_prompt_tmpl = PromptTemplate(
            #     qa_prompt_tmpl_str, function_mappings={"context_str": filter_pii_fn}
            # )

            # transform_query_engine.update_prompts(
            #     {"query_engine:response_synthesizers": qa_prompt_tmpl}
            # )
            chat_engine = index_cli.as_chat_engine( 
                chat_mode="condense_plus_context",
                # memory=llm_mem,
                system_prompt=(
                    "You are a chatbot in the receiptionist in the hospital, you should have normal interactions, as well as talk.\n" 
                    "If anything does not make sense, please inform user as well.\n"
                    "Patients will be talking to you. they may ask for recommendations of clinics for medical treatment.\n"
                    "You must suggest relevant clinics and provide office hours or opening hours For example ศูนย์ตา เปิดทำการทุกวัน เวลา 7:00-21:00 น..\n"
                    "You must explain the reasoning to your answer.\n"
                    "If your answer includes doctor name, that doctor must be drawn from the provided information. \n"
                    "Opening hours is exclusive to each clinic. Same location does not mean same opening hours.\n"
                    "Do not answer with telephone number.\n"
                    "Do not answer with same doctor name twice.\n"
                    "Do not repeat the same information twice.\n"
                    "If patient asks with symptoms and diseases, answer with relevant clinics too. \n"
                    "Prioritize the context in chat history. Continue talking about the relevant information to previous conversation.\n"
                    "If the question is a yes no question, answer yes or no then explain your reasoning. \n"
                    "You must only reference things that are provided in the context and memory\n"
                    "If you do not know , suggest ศูนย์อายุรกรรม and provide office hours or opening hours."
                    "The name of this hospital is ศิริราชปิยการุณย์\n"
                    "This hospital does not do surgical operations for beauty, except for the skin department\n"
                ),
                # text_qa_template=prompt
            )
            res = chat_engine.chat(text)#, chat_history=llm_mem.get_all())
            # res = chat_engine.query(text)
            # # res = transform_query_engine.query(text)
        else:
            data = get_appointment_from_chat(history_chat)
            print(data)
            if text == "ยืนยัน" and data != "fukng stupid":
                appointment_datetime = data["appointment_datetime"]
                duration = data["duration"]
                doctor_name = data["doctor_name"]
                patient_name = data["patient_name"]
                # doctor_name
                try:
                    doctor_id = doctor_name
                    # doctor_id = doctor_df[doctor_df["name"] == doctor_name]["doctorID"].values[0]
                    book.add_appointment(doctor_id, patientName=patient_name, datetimeStart=appointment_datetime, duration_min=duration)
                except Exception:
                    is_unbookable = True
                # except AssertionError:
                #     is_unbookable = True
        
            if isinstance(data, dict):
                doctor_name = data["doctor_name"]
                try:
                    doctor_id = doctor_df[doctor_df["name"] == doctor_name]["doctorID"].values[0]
                    appointment = book.get_appointment_by_doctor(doctor_id)
                    appointment_text = "\n".join([f'{doctor_name} has an appointment with {appoint["patientName"]}.' for appoint in appointment])
                except Exception:
                    appointment_text = ""
            chat_engine = index_cli_doc.as_chat_engine( 
                chat_mode="condense_plus_context",
                memory=llm_mem,
                system_prompt=(
                    "You are a chatbot in the receiptionist in the hospital, you should have normal interactions, as well as talk.\n" 
                    "If anything does not make sense, please inform user as well.\n"
                    f"Today is {today.strftime(book.datetimeFormat)}. You may use this information to response with user.\n"
                    "User may want to make an appointment with the doctor in some department.\n"
                    "You must advise the patient to make an appointment.\n"
                    "If the question is a yes no question, you must answer yes or no first then provide explanation to your answer.\n"
                    # "You must say the clinic name in the conversation.\n"
                    # "Prioritize the context in chat history. Continue talking about the relevant information to previous conversation.\n"
                    "Appointment may happen in the future but not in the past.\n"
                    # "You must not change or alter the time availability of each doctor.\n"
                    # "If doctor's name or doctor's time is presented, you must explain why this doctor is suitable.\n"
                    # "Do not change the availability (table_check) in weekday and time of the doctor.\n"
                    "An appointment must consist of 'appointment date time', 'doctor name', 'patient name'. Without any information, ask user kindly to fill the missing information.\n"
                    "Required: If all information for an appointment is completed, you must ask for user to confirm by saying ยืนยัน, exactly in this wording, as a form of electronic signature.\n"
                    # "Put emphasis on weekday, time and duration is low priority.\n"
                    "Every diseses is fuking cancer\n"
                    # "Each month have 5 week. The first week refers to week 1. The last week refers to week 5.\n"
                    "Do not use prior knowledge.\n"
                    # "The name of this hospital is ศิริราชปิยการุณย์\n"
                    # "This hospital does not do surgical operations for beauty, except for the skin department\n"

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
            
        res = chat_engine.chat(text, chat_history=llm_mem.get_all())
        # res = chat_engine.chat(text)
        # w = weeknum(res.response)
        # if ("สัปดาห์ที่" in res.response or "สัปดาห์" in res.response) and w is not None:
        #     eligible_days = " ".join([d.strftime(book.datetimeFormat) for d in w])
        #     previous_response = res.response
        #     chat_engine = index_cli_doc.as_chat_engine( 
        #         chat_mode="condense_plus_context",
        #         memory=llm_mem,
        #         system_prompt=(
        #             "You are an interpreter who receives text and make it eaiser to read with additional information.\n" 
        #             "use this following sentence (previous response) as a base information, {previous_response}, you must change it to be easier to understand but must also retain the same meaning.\n"
        #             "The following data is the eligible days in which week order in previous response is being represented, {eligible_days}.\n"
        #             "You must not do anything other than changing the sentence without it losing meaning, except for the case where the chosen date is not included in the eligible days, then you can ask the patient to choose a new eligible date.\n"
        #             "The patient cannot make an appointment if the date is not included in the eligible days.\n"
        #             "User just appoint the time where the doctor is not available, please reject user kindly.\n" if is_unbookable else ""
        #             f"{appointment_text}" if isinstance(data, dict) else ""
        #             # "Do not use prior knowledge except the chat history of this current session.\n"
        #             # "You are a friendly and highly empathetic chatbot to assist patients in a hospital, you should have normal interactions as a role of a receptionist."
        #             # f"Today is {today.strftime(book.datetimeFormat)}.\n"
        #             # "Patients may want to make appointments with doctors, you must be able to suggest the patient to meet specialized doctor based on their symptoms."
        #             # "You may ask more about the symptoms if you are unsure"
        #             # "Your responses must be under your context"
        #             # "Appointments can only happen in the future and must be within the scope of the correct doctor's schedule."
        #             # "Patients can only make appointments with doctors that are included in the context"
        #             # "An appointment must consist of 'appointment date time', 'doctor name', 'patient name'. Without any information, ask user kindly to fill the missing information.\n"

        #         ),
        #         # text_qa_template=prompt
        #     )
        #     res = chat_engine.chat(previous_response, chat_history=llm_mem.get_all())
        print(res.sources)
        # print(llm_mem.get_all())
        chat_timestamp = datetime.now(DEFAULT_TZ).isoformat()
        existing_session_id[sender].chat_log.append(res.response, chat_timestamp, "chat bot")
        return jsonify({"timestamp": chat_timestamp, "text": res.response, "sender": "chat bot", "id": uuid4()}), 200

    except Exception as e:
        print(e)
        return jsonify({"message": "Invalid format", "exception_msg": str(e)}), 400

def weeknum(text, default_today=True):
    text = text.split(" ")
    num = ""
    for i, t in enumerate(text):
        if t == "สัปดาห์ที่":
            num = text[i+1]
            break
    if num != "":
        num = num.split(",")
    # if default_today:
    all_days = []
    for n in num: # 1 3 5
        try:
            n = int(n)
        except:
            continue
        range_start = (1 + (7*(n-1)))
        range_end = range_start + 7
        # n = int(n)
        for r in range(range_start, range_end+1):
            print( str(r), parse("day " + str(r) + "of next month"))
            all_days.append(parse("next month, day " + str(r)))

    return all_days


    

if __name__ == "__main__":
    book.create_appoint_log()
    port = os.getenv("PORT")
    app.run("0.0.0.0", port, debug=True)
