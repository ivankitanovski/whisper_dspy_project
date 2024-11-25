import dspy

from brain.signatures.analytics import AnalyticsSignature


class AnalyticsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.TypedPredictor(AnalyticsSignature)

    def forward(
        self,
        chat_history: dict,
    ):
        return self.prog(chat_history=chat_history)
