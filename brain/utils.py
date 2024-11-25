import json

import dspy

from brain.models import ChatHistory, ChatMessage


def load_data(path):
    """Utility function to load the data and return in the dspy.Example format."""
    with open(path) as f:
        data = json.load(f)

    result = []
    for obj in data:
        # create dspy.Example object
        sample = dspy.Example(
            chat_history=ChatHistory(**obj["chat_history"]), output=obj["output"]
        ).with_inputs("chat_history")

        result.append(sample)

    return result
