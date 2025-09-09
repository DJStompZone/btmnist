"""
hf_hub.py — lightweight helpers for pushing checkpoints to Hugging Face Hub.

This intentionally avoids touching README.md unless explicitly asked, and
handles repo creation on first push. Authentication is via the standard
HUGGINGFACE_HUB_TOKEN env var or prior `huggingface-cli login`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Tuple

from huggingface_hub import (
    HfApi,
    create_repo,
    upload_file,
    whoami,
    hf_hub_url,
)
from huggingface_hub._commit_api import CommitOperationAdd


@dataclass
class HubSpec:
    """Configuration for where/how to push artifacts."""
    repo_id: str
    repo_type: str = "model"
    private: bool = False
    branch: Optional[str] = None
    commit_message: str = "btmnist: update checkpoint"
    create_ok: bool = True


def ensure_repo(spec: HubSpec) -> None:
    """
    Ensure the target repo exists. Creates it if missing and allowed.

    Raises if the user is not authenticated and the repo is under their namespace.
    """
    api = HfApi()
    try:
        _ = whoami()
    except Exception as e:
        raise RuntimeError(
            "Not authenticated to Hugging Face. Run `huggingface-cli login` "
            "or set HUGGINGFACE_HUB_TOKEN."
        ) from e

    try:
        api.repo_info(spec.repo_id, repo_type=spec.repo_type)
        return
    except Exception:
        if not spec.create_ok:
            raise

    owner, _, name = spec.repo_id.partition("/")
    if not name:
        raise ValueError(f"Invalid repo_id: {spec.repo_id}")
    create_repo(
        repo_id=spec.repo_id,
        private=spec.private,
        repo_type=spec.repo_type,
        exist_ok=True,
    )


def push_file(local_path: str, remote_path: str, spec: HubSpec) -> str:
    """
    Upload a single file to the Hub; returns the blob URL.
    """
    url = upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        revision=spec.branch,
        commit_message=spec.commit_message,
    )
    return url


def push_many(pairs, spec: HubSpec) -> None:
    """
    Commit many files in one shot. Only changed files are included
    """
    api = HfApi()
    ops = []
    for local, remote in pairs:
        with open(local, "rb") as f:
            data = f.read()
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=data))
    api.create_commit(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        operations=ops,
        commit_message=spec.commit_message,
        revision=spec.branch,
    )


def dump_json(obj, path: str) -> None:
    """Write a compact JSON file to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
