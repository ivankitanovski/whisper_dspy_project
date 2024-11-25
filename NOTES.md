# Assignment notes

This document provides an overview of my thought process, the steps I followed during the implementation, and additional context regarding the decisions I made while completing the assignment. I will cover it for each task separately.

1. **Improve Client Personality Emulation**

   To complete this task, I did the following steps:

   1. Added utility function in `brain.utils.load_data` to load the data and convert it to `dsp.Example` type.
   2. Added a `can_optimize` method in the `brain.modules.chatter.ChatterModule`. This method checks whether there is training data to do any optimization. The data should be passed in the `examples` parameter of the constructor.
   3. Added `optimize` method in the `brain.modules.chatter.ChatterModule`. The method runs the `KNNFewShot` optimizer and replaces the current responder object with the compiled model, so it can be used later on.
   4. The compiled model is saved for future reuse.
   5. The module uses the compiled model in the main app.

   This can be further tuned by adding custom metric and, potentially, adding a separate teacher module.

2. **Incorporate Context Awareness**

   To improve context awareness, I added a `timestamp` in the `brain.models.ChatMessage`. The timestamp is incorporated in the message and the model can react based on this. If there is a need for additional data points to be used in the context, we can add a separate meta model containing such data and embed it in the message.

3. **Topic Filtering**

   I extended the prompt in the `brain.signatures.responder.ResponderSignature` to take note of the constraints. During tests it worked OK. This can be extended into a separate filter module if the contraints become more specific.

4. **Further Product Enhancements**

   I added a `brain.signatures.analytics.AnalyticsSignature` and, corresponding, `brain.modules.analytics.AnaltyicsSignature` to keep track of the current buying interest level of the fan. Currently, it's just a string explaining the interest. This can be further tuned into concrete categories for easier processing and analysis.

**Other notes/improvements**

- Reorganized `chat_interface.py` to accept CLI parameters. The possible parameters are `seed` and `examples`
  - `seed` - This is the seed used for the randomness through out the project. Especially, important to have reproducable results (as much as possible) during testing/exprimenting. If omitted, it will not be set.
    - E.g. `python brain/chat_interface.py --seed 42`
  - `examples` - The path to the traing set. This is used to pass trainging data to the chat interface. If not passed, it will not try to optimize the model.
    - E.g. `python brain/chat_interface.py --examples training_data/conversations.json`
- Added `pre-commit` hook. I like to have some basic structure in the code and have that through out the repo.
- Updated `pyproject.toml` to reflect the new packagse that are needed for whole project to run.

## How to run

Make sure to install additional requirements as the optmizers required (`poetry install`). To run the app you just need to:

```shell
python brain/chat_interface.py --examples training_data/conversations.json
```

## References

These are the resources, I used to inform my implementation:

- https://dspy.ai/learn/ - Went through the entire documentation here.
- https://github.com/stanfordnlp/dspy/tree/main/examples - Went through appropriate examples from their repo.
- https://github.com/stanfordnlp/dspy - The code is the source of truth. Finally, for any dilemas, I just read through how the code is actually implemented.
- Also, a few short youtube overviews of the framework: https://www.youtube.com/@insightbuilder/videos,

## Final note

I knew of DSPy, but never had a chance to get hands-on before, so had to get myself familiar with the framework. It took me more than the expected 2 hours, probably around 5-6 hours end-to-end: reading, researching, coding, experimenting. Plus, ~1hr writing this doc :smile:
