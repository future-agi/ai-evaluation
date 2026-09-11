"""Strict parsing for public GitHub repository and branch URLs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import unquote, urlsplit


_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_REF = re.compile(r"^[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class GitHubLocation:
    repository: str
    ref: str | None = None

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.repository}.git"


def parse_github_location(raw: str) -> GitHubLocation:
    """Parse ``owner/repo`` or a public GitHub repository/branch URL.

    A ``/tree/<ref>`` link is treated as a branch link. The entire suffix is retained, so branch
    names such as ``feat/harness`` work without a separate ref field. File, commit, pull-request,
    and arbitrary GitHub page URLs are deliberately rejected.
    """
    value = raw.strip()
    if not value:
        raise ValueError("github_repository_invalid")

    if "://" not in value:
        pieces = value.removesuffix(".git").strip("/").split("/")
        if len(pieces) != 2 or not all(_NAME.fullmatch(item) for item in pieces):
            raise ValueError("github_repository_invalid")
        return GitHubLocation(repository="/".join(pieces))

    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("github_repository_invalid")

    pieces = [unquote(item) for item in parsed.path.strip("/").split("/") if item]
    if len(pieces) < 2:
        raise ValueError("github_repository_invalid")
    owner, repository = pieces[:2]
    repository = repository.removesuffix(".git")
    if not _NAME.fullmatch(owner) or not _NAME.fullmatch(repository):
        raise ValueError("github_repository_invalid")

    if len(pieces) == 2:
        return GitHubLocation(repository=f"{owner}/{repository}")
    if len(pieces) < 4 or pieces[2] != "tree":
        raise ValueError("github_repository_url_must_point_to_repository_or_branch")

    ref = "/".join(pieces[3:]).strip("/")
    if not ref or ".." in ref or not _REF.fullmatch(ref):
        raise ValueError("github_ref_invalid")
    return GitHubLocation(repository=f"{owner}/{repository}", ref=ref)


__all__ = ["GitHubLocation", "parse_github_location"]
