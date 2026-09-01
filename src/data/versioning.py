from hashlib import sha256
from pathlib import Path


def fingerprint_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> dict[str, str | int]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    path = Path(path)
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)

    return {
        "path": str(path),
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }
