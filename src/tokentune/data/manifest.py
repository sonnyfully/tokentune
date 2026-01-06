from future import annotations

import json
from pathlib import Path
from typing import Any

def write_manifest(path: Path | str, record: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")