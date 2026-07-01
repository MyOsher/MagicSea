"""Load the project config (config.yaml) as a plain dict."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml from the project root (or an explicit path)."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to the project root so scripts work from anywhere.
    for key, rel in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(ROOT / rel)
    return cfg


if __name__ == "__main__":
    import json
    print(json.dumps(load_config(), indent=2))
