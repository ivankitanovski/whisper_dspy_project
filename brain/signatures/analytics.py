from typing import Literal

import dspy

from brain.models import ChatHistory


class AnalyticsSignature(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    Your job is to analyze the chat history and determine the buying interest level of the FAN.
    """

    chat_history: ChatHistory = dspy.InputField(desc="the chat history")
    buying_interest_level: str = dspy.OutputField(
        desc="the user's buying interest level",
    )
