"""Configure Hugging Face Space runtime variables and secrets."""

import importlib
import os
import sys
from pathlib import Path


def load_dotenv_if_present() -> tuple[bool, Path]:
    """Load .env values without overriding existing shell exports."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return False, env_path

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True, env_path


def require_env(name: str, dotenv_found: bool, env_path: Path) -> str:
    value = os.getenv(name)
    if not value:
        if dotenv_found:
            raise RuntimeError(
                f"Missing required environment variable: {name}. "
                f"Please update {env_path} with real values and re-run."
            )
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"{env_path} was not found. Create it or export variables in your shell."
        )

    lowered = value.strip().lower()
    if lowered.startswith("your_") or "token_used_by_model_calls" in lowered:
        raise RuntimeError(
            f"Environment variable {name} looks like a placeholder value. "
            f"Please update {env_path} (or exported env vars) with real tokens/IDs."
        )
    return value


def main() -> None:
    dotenv_found, env_path = load_dotenv_if_present()

    hf_token = require_env("HF_TOKEN", dotenv_found, env_path)
    repo_id = require_env("SPACE_REPO_ID", dotenv_found, env_path)
    inference_hf_token = require_env("INFERENCE_HF_TOKEN", dotenv_found, env_path)

    try:
        hf_module = importlib.import_module("huggingface_hub")
        hf_api_cls = getattr(hf_module, "HfApi")
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    api = hf_api_cls(token=hf_token)

    # Non-sensitive vars
    api.add_space_variable(
        repo_id=repo_id,
        key="API_BASE_URL",
        value="https://router.huggingface.co/v1",
    )
    api.add_space_variable(
        repo_id=repo_id,
        key="MODEL_NAME",
        value="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Sensitive var
    api.add_space_secret(
        repo_id=repo_id,
        key="HF_TOKEN",
        value=inference_hf_token,
    )

    print("Space variables/secrets configured.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Setup aborted: {exc}")
        print("Required keys: HF_TOKEN, SPACE_REPO_ID, INFERENCE_HF_TOKEN")
        sys.exit(1)
