"""Example configuration for AstroBench.

Prefer environment variables for credentials. Do not commit real API keys.
"""

OPENAI_API_KEY = ""
OPENAI_BASE_URL = None
OPENAI_USE_AZURE = False
OPENAI_AZURE_ENDPOINT = ""
OPENAI_API_VERSION = ""
OPENAI_JUDGE_MODEL = ""

# For OpenAI-compatible second-judge endpoints, set these in your shell:
# export JUDGE_OPENAI_API_KEY="..."
# export JUDGE_OPENAI_BASE_URL="..."
# export SECOND_JUDGE_MODEL="claude-sonnet-4-6"
