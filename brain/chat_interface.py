import argparse
import random
from datetime import datetime

import dspy
import numpy as np
import torch

from brain.lms.together import Together
from brain.models import ChatHistory, ChatMessage
from brain.modules.chatter import ChatterModule
from brain.utils import load_data


def set_seed(seed):
    """Set seed for reproducibility. It's important to have reproducible results (as much as possible) for testing and experimenting!"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Whisper chatbot interface.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed value for reproducibility (omit for randomness).",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--examples",
        type=str,
        help="Path to the json file where the training data is located.",
        required=False,
        default=None,
    )
    return parser.parse_args()


def main(seed, examples):
    """Main script to run the app."""
    # set seed at the beginning of the script
    set_seed(seed)

    # configure the language model
    lm = Together(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        temperature=0.5,
        max_tokens=1000,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.2,
        stop=["<|eot_id|>", "<|eom_id|>", "\n\n---\n\n", "\n\n---", "---", "\n---"],
        # stop=["\n", "\n\n"],
    )
    dspy.settings.configure(lm=lm)

    # load examples
    data = load_data(examples) if examples else None

    # initialize chat history and chatter module
    chat_history = ChatHistory()
    chatter = ChatterModule(examples=data)
    if chatter.can_optimize():
        chatter.optimize()
        chatter.save("compiled_llms/chatter.json")  # save it for later use and testing

    # start chat
    while True:
        # get user input
        user_input = input("You: ")

        # append user input to chat history
        chat_history.messages.append(
            ChatMessage(
                from_creator=False,
                content=user_input,
                timestamp=datetime.now(),
            ),
        )

        # send request to endpoint
        response = chatter(chat_history=chat_history).output

        # append response to chat history
        chat_history.messages.append(
            ChatMessage(
                from_creator=True,
                content=response,
                timestamp=datetime.now(),
            ),
        )
        # print response
        print("---------------------------------------")
        print("Response:", response)
        print("---------------------------------------")
        # uncomment this line to see the
        # lm.inspect_history(n=1)

        # print buying interest
        print("***************************************")
        print("Current buying interest level:", chatter.buying_interest_level)
        print("***************************************")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.seed, args.examples)
