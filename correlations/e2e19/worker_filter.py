"""Load the publication-time E2E annotator exclusion set.

The public artifact bundle uses pseudonymous worker identifiers. Raw platform
worker IDs are intentionally not stored in the code repository.
"""

from collections.abc import Iterator, Set
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().parent / "human_ratings" / "excluded_workers.txt"


class ExcludedWorkers(Set[str]):
    def __init__(self, path: Path):
        self.path = path
        self._values: frozenset[str] | None = None

    def _load(self) -> frozenset[str]:
        if self._values is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"Missing {self.path}. Download the de-identified E2E "
                    "human-ratings release linked from the repository README."
                )
            self._values = frozenset(
                line.strip()
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        return self._values

    def __contains__(self, value: object) -> bool:
        return value in self._load()

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


def load_excluded_workers(path: Path = DEFAULT_PATH) -> ExcludedWorkers:
    return ExcludedWorkers(path)
