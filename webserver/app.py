import os
from flask import Flask, request, jsonify

from datetime import datetime, timezone, timedelta
from uuid import uuid4

from get_intent import get_intent_from_chat
from SessionMemory import SessionMemory

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
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

index = get_index(os.getenv("MERGING_INDEX_DIR"))

app = Flask(__name__)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/chatquery", methods=["POST"])
def chat_query():
    try:
        data = request.json
        text = data["text"]
        sender = data["sender"]
        timestamp = data["timestamp"]
        _id = data["id"]

        if not(isinstance(text, str) and isinstance(sender, str) and isinstance(_id, str)):
            raise Exception("type of request body is not right")
        # 1. find intent
        intent = get_intent_from_chat(text)
        # havent do anythign with intent
        
        # 1.5 retrieve existing session if exists
        session = existing_session_id.get(sender)
        if session:
            llm_mem = session.llm_memory
        else:
            llm_mem = ChatMemoryBuffer.from_defaults(token_limit=1500)
            existing_session_id[sender] = SessionMemory(sender, llm_mem)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=llm_mem,
            system_prompt=(
                "You are a chatbot in the receiptionist in the hospital, you should have normal interactions, as well as talk. Do not hallucinate."
            ),
        )

        # 2. 
        existing_session_id[sender].chat_log.append(text, timestamp, sender)
        res = chat_engine.chat(text)
        chat_timestamp = datetime.now(DEFAULT_TZ).isoformat()
        existing_session_id[sender].chat_log.append(text, chat_timestamp, "chat engine")
        return jsonify({"intent": intent, "timestamp": chat_timestamp, "text": res.response, "sender": "chat bot", "id": uuid4()}), 200

    except Exception as e:
        print(e)
        return jsonify({"message": "Invalid format", "exception_msg": str(e)}), 400


if __name__ == "__main__":
    port = os.getenv("PORT")
    app.run("0.0.0.0", port, debug=True)
