import json
from pathlib import Path
from typing import Any


class JsonlStore:
    '''
    store historical questions and evaluations in jsonl format
    '''
    def __init__(self, path: str) -> None:
        """Initialize a JSONL store and ensure the file exists."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, item: dict[str, Any]) -> None:
        '''
        write to file
        '''
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        """Load all JSONL records from the store."""
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
