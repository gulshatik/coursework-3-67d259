# LangGraph Agent with Memory and Confirmation

This repository contains a minimal example of a LangGraph agent that:

1. **Remembers** the conversation history across turns using `MemorySaver`.
2. **Pauses** before invoking any tool, allowing the user to confirm the action.
3. Uses the `rich` library for pretty console output.

The agent is demonstrated with a simple `get_price` tool that returns a mock price for a city on a given date.  
The script runs without a TTY – it uses a predefined list of questions to showcase the behaviour.

## Setup

```bash
pip install -r requirements.txt
```

Make sure you have a `.env` file with the `BROJS_PAT_TOKEN` variable set to your API key.

## Run

```bash
python main.py
```

You will see the agent’s responses, the tool calls, and the confirmation prompts printed in the console.
