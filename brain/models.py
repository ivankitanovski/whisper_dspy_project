from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    from_creator: bool
    content: str
    timestamp: Optional[datetime] = None # can extend to separate meta object if we need to capture more data here

    def __str__(self):
        role = "YOU" if self.from_creator else "THE FAN"
        if self.timestamp:
            message = f"{role} [{self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}]: {self.content}"
        else:
            message = f"{role}: {self.content}"
        return message

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = []

    def __str__(self):
        messages = []
        for i, message in enumerate(self.messages):
            message_str = str(message)
            # if i == len(self.messages) - 1 and not message.from_creator:
            #     message_str = (
            #         "(The fan just sent the following message which your message must respond to): "
            #         + message_str
            #     )
            messages.append(message_str)
        return "\n".join(messages)

    def model_dump_json(self, **kwargs):
        return str(self)
