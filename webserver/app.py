import os
from flask import Flask, request, jsonify
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from get_intent import get_intent_from_chat

from llama_index.core.memory import ChatMemoryBuffer, BaseMemory
from llama_index.core import VectorStoreIndex
from llama_index.legacy import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex)
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ChatInstance:
    timestamp: datetime
    msg: str
    author: str

@dataclass
class ChatLog:
    log:list[ChatInstance] = field(default_factory=list)
    def append(self, msg, timestamp, author):
        self.log.append(ChatInstance(timestamp, msg, author))

@dataclass
class SessionMemory:
    sender: str
    llm_memory: BaseMemory
    chat_log: ChatLog = field(default_factory=ChatLog)


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
            raise TypeError()
        
        session = existing_session_id.get(sender)
        if session:
            llm_mem = session.llm_memory

        else:
            llm_mem = ChatMemoryBuffer.from_defaults(token_limit=1500)
            existing_session_id[_id] = SessionMemory(_id, llm_mem)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=llm_mem,
            system_prompt=(
                "You are a chatbot, able to have normal interactions, as well as talk"
                " about an essay discussing Paul Grahams life."
            ),
        )

        # response = chat_engine.chat("Hello!")

        # 1. find intent
        intent = get_intent_from_chat(text)
        # havnt do anythign with intent

        # 2. 
        existing_session_id[_id].chat_log.append(text, timestamp, sender)
        res = chat_engine.chat(text)
        chat_timestamp = datetime.now(DEFAULT_TZ).isoformat()
        existing_session_id[_id].chat_log.append(text, chat_timestamp, "chat engine")
        return jsonify({"intent": intent, "timestamp": chat_timestamp, "message": res.response, "sender": "Chat Bot", "id": uuid4()})

    except Exception as e:
        print(e.with_traceback())
        return jsonify({"message": "Invalid format", "exception_msg": str(e)})
    return "<p>Hello, World!</p>"


if __name__ == "__main__":
    app.run("0.0.0.0", "8876", debug=True)
