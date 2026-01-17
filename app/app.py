"""
app.py - Streamlit Web Interface for Repository Q&A Agent

This module provides an interactive chat interface for querying repository
documentation using a PydanticAI agent. Users can ask questions and receive
AI-generated answers with citations to source files.

Application Flow:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         STREAMLIT WEB APP                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
    │   │  User Input  │───▶│  Agent Run   │───▶│  Streamed Response   │  │
    │   │  (Chat Box)  │    │  (Async)     │    │  (Real-time UI)      │  │
    │   └──────────────┘    └──────────────┘    └──────────────────────┘  │
    │          │                   │                      │               │
    │          │                   ▼                      │               │
    │          │           ┌──────────────┐               │               │
    │          │           │ Search Tool  │               │               │
    │          │           │ (minsearch)  │               │               │
    │          │           └──────────────┘               │               │
    │          │                   │                      │               │
    │          ▼                   ▼                      ▼               │
    │   ┌────────────────────────────────────────────────────────────┐   │
    │   │                    Session State                            │   │
    │   │    messages: [{role: "user", content: "..."}, ...]         │   │
    │   └────────────────────────────────────────────────────────────┘   │
    │                              │                                      │
    │                              ▼                                      │
    │                    ┌──────────────────┐                            │
    │                    │   logs.py        │                            │
    │                    │   (JSON files)   │                            │
    │                    └──────────────────┘                            │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # Run the Streamlit app
    $ cd app
    $ streamlit run app.py

    # Or with uv
    $ uv run streamlit run app/app.py

Environment Variables:
    OPENAI_API_KEY: Required for LLM access
    REPO_OWNER: GitHub org/user (default: "ethereum")
    REPO_NAME: Repository name (default: "EIPs")
    MODEL_NAME: LLM model (default: "gpt-4o-mini")
"""

import os
import asyncio
import streamlit as st

import ingest
import search_agent
import logs


def secret_or_env(key: str, default=None):
    """
    Get configuration value from Streamlit secrets or environment variables.

    Streamlit Cloud stores secrets in .streamlit/secrets.toml, but local
    development may use environment variables. This helper tries secrets
    first, then falls back to environment variables.

    Args:
        key: Configuration key name
        default: Default value if not found in either location

    Returns:
        Configuration value or default

    Example:
        >>> api_key = secret_or_env("OPENAI_API_KEY")
        >>> repo = secret_or_env("REPO_OWNER", "ethereum")
    """
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        # Fall back to environment variables (for local development)
        return os.getenv(key, default)


# ============================================================================
# CONFIGURATION
# ============================================================================
# These can be overridden via app/.streamlit/secrets.toml or environment vars

REPO_OWNER = secret_or_env("REPO_OWNER", "ethereum")
REPO_NAME = secret_or_env("REPO_NAME", "EIPs")

# Note: Branch is auto-detected by ingest.py (tries "main" then "master")
# Uncomment to force a specific branch:
# BRANCH = secret_or_env("REPO_BRANCH", None)


# ============================================================================
# AGENT INITIALIZATION (Cached)
# ============================================================================

@st.cache_resource
def init_agent_cached():
    """
    Initialize the agent and search index (runs once per session).

    This function is cached using Streamlit's cache_resource decorator,
    meaning it only runs once when the app starts or is deployed.
    Subsequent requests reuse the cached agent and index.

    The initialization process:
    1. Downloads repository from GitHub as ZIP
    2. Parses markdown files with YAML frontmatter
    3. Chunks documents for better retrieval
    4. Builds minsearch index
    5. Creates PydanticAI agent with search tool

    Returns:
        Tuple of (agent, branch) where:
            - agent: Configured PydanticAI Agent
            - branch: Git branch that was indexed
    """
    st.write("Indexing repository (runs once per session/deployment)...")

    # Build searchable index from repository
    # - chunk=True: Split documents into overlapping chunks
    # - size=2000: Each chunk is up to 2000 characters
    # - step=1000: 50% overlap between chunks
    index, branch = ingest.index_data(
        REPO_OWNER,
        REPO_NAME,
        branches=("main", "master"),
        chunk=True,
        chunking_params={"size": 2000, "step": 1000},
    )

    # Create agent with search tool connected to index
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME, branch)

    return agent, branch


# Initialize agent (uses cache after first run)
agent, branch = init_agent_cached()


# ============================================================================
# STREAMLIT UI SETUP
# ============================================================================

st.set_page_config(page_title="EIP Specs Agent", layout="centered")
st.title("EIP Specs Agent")
st.caption(f"Ask questions about {REPO_OWNER}/{REPO_NAME} (branch: {branch})")

# Initialize chat history in session state
# This persists across reruns within the same browser session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================================
# STREAMING RESPONSE HANDLER
# ============================================================================

def run_streaming_answer(prompt: str) -> str:
    """
    Execute agent query with streaming output.

    Uses asyncio to run the PydanticAI agent's streaming interface.
    Updates the Streamlit UI in real-time as tokens are generated.

    Technical Notes:
    - Uses asyncio.run() to handle async agent calls in sync Streamlit context
    - Employs st.empty() placeholder for incremental UI updates
    - Debounces output by 0.02s to avoid overwhelming the UI
    - Logs interaction to file after completion

    Args:
        prompt: User's question

    Returns:
        Complete response text

    Example:
        >>> response = run_streaming_answer("What is ERC-20?")
        >>> print(response)  # Full answer with citations
    """
    # Create placeholder for streaming updates
    placeholder = st.empty()
    full_text = ""

    async def _run():
        """Inner async function for agent execution."""
        nonlocal full_text

        # Stream response from agent
        async with agent.run_stream(user_prompt=prompt) as result:
            # Iterate over streamed chunks with debouncing
            async for chunk in result.stream_output(debounce_by=0.02):
                full_text = chunk
                # Update UI with accumulated text
                placeholder.markdown(full_text)

            # Log the complete interaction after streaming finishes
            logs.log_interaction_to_file(agent, result.new_messages())

    # Run async function in sync context
    asyncio.run(_run())

    return full_text


# ============================================================================
# CHAT INPUT HANDLER
# ============================================================================

# Chat input widget (appears at bottom of page)
if prompt := st.chat_input("Ask your question..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_text = run_streaming_answer(prompt)

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": final_text})
