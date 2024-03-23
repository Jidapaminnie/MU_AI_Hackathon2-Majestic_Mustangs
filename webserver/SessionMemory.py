from dataclasses import dataclass, field
from datetime import datetime
from llama_index.core.memory import BaseMemory



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
