"""Shared Groq LLM client module for LinkedIn Post Generator."""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import config

# Ensure .env is loaded
load_dotenv()


def get_llm(
    model: Optional[str] = None,
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> ChatGroq:
    """Initialize and return a ChatGroq client.

    Args:
        model: Optional model name override. Defaults to GROQ_MODEL env var
               or config.DEFAULT_MODEL.
        temperature: Sampling temperature for generation.

    Returns:
        Configured ChatGroq instance.

    Raises:
        ValueError: If GROQ_API_KEY is missing or empty.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY is missing or empty. Please set GROQ_API_KEY in your .env file."
        )

    model_name = model or os.environ.get("GROQ_MODEL") or config.DEFAULT_MODEL

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )


def update_groq_model_in_env(new_model: str) -> None:
    """Update GROQ_MODEL in .env preserving all other keys (including GROQ_API_KEY)."""
    env_path = config.BASE_DIR / ".env"
    if not env_path.exists():
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"GROQ_MODEL={new_model}\n")
        return

    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    model_found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("GROQ_MODEL="):
            new_lines.append(f"GROQ_MODEL={new_model}\n")
            model_found = True
        else:
            new_lines.append(line)

    if not model_found:
        new_lines.append(f"\nGROQ_MODEL={new_model}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def smoke_test() -> None:
    """Run smoke test against Groq LLM and recover/select model if needed."""
    load_dotenv(override=True)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        print("ERROR: GROQ_API_KEY is missing or empty in .env. Please configure it.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("GROQ_MODEL", config.DEFAULT_MODEL)
    print(f"Testing model: {model}...")

    try:
        llm = get_llm(model=model)
        response = llm.invoke("Hello, respond with 'OK'.")
        print(f"Smoke test succeeded with model '{model}'! Response: {response.content.strip()[:60]}")
    except Exception as e:
        print(f"Model '{model}' failed with error: {e}")
        print("Querying Groq API for available models...")
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            models = client.models.list()
            active_models = [m.id for m in models.data if getattr(m, "active", True)]
            print(f"Available active models ({len(active_models)}):")
            for m_id in sorted(active_models):
                print(f" - {m_id}")

            # Prefer general text LLMs like llama-3.1-8b-instant, mixtral, or gemma
            preferred_candidates = [
                "llama-3.1-8b-instant",
                "llama3-8b-8192",
                "llama-3.3-70b-versatile",
                "gemma2-9b-it",
            ]
            chosen = None
            for cand in preferred_candidates:
                if cand in active_models:
                    chosen = cand
                    break

            if not chosen and active_models:
                # Pick first model that isn't whisper
                for m_id in active_models:
                    if "whisper" not in m_id.lower() and "guard" not in m_id.lower():
                        chosen = m_id
                        break

            if not chosen:
                chosen = active_models[0]

            print(f"Selecting replacement model: {chosen}")
            update_groq_model_in_env(chosen)
            os.environ["GROQ_MODEL"] = chosen

            # Retry smoke test with new model
            new_llm = get_llm(model=chosen)
            test_resp = new_llm.invoke("Hello, respond with 'OK'.")
            print(f"Smoke test succeeded with new model '{chosen}'! Response: {test_resp.content.strip()[:60]}")
        except Exception as recovery_err:
            print(f"Failed to recover model: {recovery_err}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    smoke_test()
