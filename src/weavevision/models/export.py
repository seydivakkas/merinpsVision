"""Model artifact hashing and atomic metadata helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_artifact(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 without unsafe deserialization.

    OpenVINO IR uses an XML graph and sibling BIN weights. Both files, including their names
    and lengths, are included in one combined identity so registry verification cannot accept
    a modified weight companion.
    """
    digest = hashlib.sha256()
    members = [path]
    if path.suffix.casefold() == ".xml":
        binary = path.with_suffix(".bin")
        if not binary.is_file():
            raise FileNotFoundError(f"OpenVINO weight companion missing: {binary}")
        members.append(binary)
    for member in members:
        encoded_name = member.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(member.stat().st_size.to_bytes(8, "big"))
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    return digest.hexdigest()
