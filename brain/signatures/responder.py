import dspy

from brain.models import ChatHistory


class ResponderSignature(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    You are deciding on what your message should be.

    Notes:
    - You responses should avoid mentioning or referencing any social platforms other than OnlyFans.
    - Refrain from suggesting or discussing in-person meetings with fans.
    """

    chat_history: ChatHistory = dspy.InputField(desc="the chat history")

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )
