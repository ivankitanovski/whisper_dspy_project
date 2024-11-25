from typing import Optional

import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import KNNFewShot

from brain.models import ChatHistory
from brain.modules.analytics import AnalyticsModule
from brain.modules.responder import ResponderModule


class ChatterModule(dspy.Module):
    def __init__(self, examples: Optional[dict]):
        super().__init__()
        self.responder = (
            ResponderModule()
        )  # filtering could be done with a separate module as well
        self.analytics = AnalyticsModule()
        self.examples = examples
        self.buying_interest_level = None

    def can_optimize(self):
        return self.examples is not None and len(self.examples) > 0

    def optimize(self):
        if self.can_optimize():
            # optimize the model
            teleprompter = KNNFewShot(
                k=min(7, len(self.examples)), trainset=self.examples
            )  # can experiment with k here
            self.responder = teleprompter.compile(self.responder)
        else:
            raise ValueError("No examples provided to train on.")

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        self.buying_interest_level = self.analytics(
            chat_history=chat_history
        ).buying_interest_level
        return self.responder(chat_history=chat_history)
