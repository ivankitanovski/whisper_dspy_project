import dspy
from signatures.responder import ResponderSignature

from brain.models import ChatHistory


class ResponderModule(dspy.Module):
    def __init__(self):
        super().__init__()
        reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step to decide on our message. We",
        )
        self.prog = dspy.TypedChainOfThought(ResponderSignature, reasoning=reasoning)

    def forward(
        self,
        chat_history: dict,
    ):
        return self.prog(
            chat_history=ChatHistory.model_validate(chat_history),
        )
