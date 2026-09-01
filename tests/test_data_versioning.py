from hashlib import sha256

import pytest

from src.data.versioning import fingerprint_file


def test_fingerprint_file_streams_sha256_metadata(tmp_path):
    data_path = tmp_path / "ratings.csv"
    payload = b"userId,movieId,rating\n1,2,4.5\n"
    data_path.write_bytes(payload)

    fingerprint = fingerprint_file(data_path, chunk_size=3)

    assert fingerprint == {
        "path": str(data_path),
        "algorithm": "sha256",
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_fingerprint_file_rejects_invalid_chunk_size(tmp_path):
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        fingerprint_file(tmp_path / "ratings.csv", chunk_size=0)
