import base64
import os
from typing import Optional, ClassVar, Set
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator

# Keep this class minimal & fast: no network calls; only local file -> data URI.
# The backend will detect input type and do any heavy lifting.

class ProtectInputAdapter(BaseModel):
    """
    Minimal, production-safe input wrapper for Protect.

    - Accepts text, http(s) URLs, data URIs, and local audio/image files.
    - Rejects known HTML 'viewer' URLs (e.g., GitHub blob) to avoid 400s downstream.
    - Converts local media files to 'data:' URIs (cheap; no networking).
    - Leaves type detection to the backend.
    """
    input: str
    call_type: str = "protect"

    AUDIO_EXTENSIONS: ClassVar[Set[str]] = {
        ".mp3", ".wav",
    }
    IMAGE_EXTENSIONS: ClassVar[Set[str]] = {
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"
    }
    # Optional cap for local files (bytes). Keep small to protect latency/footguns.
    MAX_LOCAL_BYTES: ClassVar[int] = int(os.getenv("FI_PROTECT_MAX_LOCAL_BYTES", "20000000"))  # 20MB

    # --- Pydantic v2: run after parsing ---
    def model_post_init(self, __context) -> None:
        try:
            s = (self.input or "").strip()
            if not s:
                raise ValueError("Input cannot be empty or whitespace")

            # 1) Data URIs: validate & allow
            if s.startswith("data:"):
                mime = s.split(";", 1)[0].split(":", 1)[-1].lower()
                if not (mime.startswith("audio/") or mime.startswith("image/")):
                    # Treat non-media data URIs as text – Protect supports only text/image/audio.
                    raise ValueError("Unsupported data URI mime; only audio/* or image/* allowed")
                # Pass as-is
                return

            p = urlparse(s)

            # 2) HTTP(S) URL: allow pass-through, but block known viewer pages
            if p.scheme in ("http", "https"):
                if self._is_blocked_url(s, p):
                    raise ValueError(
                        "This link looks like a preview page, not a direct file. "
                        "Use a direct download URL (e.g., raw.githubusercontent.com for GitHub; "
                        "export=download for Google Drive; dl.dropboxusercontent.com for Dropbox)."
                    )
                # Do not sniff/transform: backend will handle it
                return

            # 3) Local file path → convert to data URI (fast, no network)
            if self._looks_like_local_path(s):
                ext = os.path.splitext(s)[1].lower()
                if ext in self.AUDIO_EXTENSIONS:
                    self.input = self._file_to_data_uri(s, self._audio_mime(ext))
                    return
                if ext in self.IMAGE_EXTENSIONS:
                    self.input = self._file_to_data_uri(s, self._image_mime(ext))
                    return
                # Not a supported media extension
                allowed_audio = ", ".join(sorted(self.AUDIO_EXTENSIONS))
                allowed_img = ", ".join(sorted(self.IMAGE_EXTENSIONS))
                raise ValueError(
                    f"Unsupported local file type '{ext}'. Supported audio: {allowed_audio}. "
                    f"Supported image: {allowed_img}."
                )

            # 4) Otherwise: treat as plain text (backend handles as text)
            # Nothing to change.
        except Exception as e:
            # Fail fast with a clean message. Protect supports only text/image/audio.
            raise ValueError(f"Invalid Protect input: {e}") from e

    @staticmethod
    def _looks_like_local_path(s: str) -> bool:
        # Absolute or relative file path that exists
        try:
            return os.path.exists(s)
        except Exception:
            return False

    @staticmethod
    def _is_blocked_url(s: str, p) -> bool:
        host = (p.netloc or "").lower()
        path = (p.path or "").lower()

        # Known viewer / HTML pages that won't return raw bytes:
        # GitHub blob pages
        if host == "github.com" and "/blob/" in path:
            return True
        # Google Drive viewers
        if host in ("drive.google.com", "docs.google.com") and ("/file/" in path or "/uc" in path) and "export=download" not in s:
            return True
        # Dropbox share pages (use dl.dropboxusercontent.com for direct)
        if host == "www.dropbox.com" and "/s/" in path:
            return True
        # OneDrive viewer links
        if "onedrive.live.com" in host and "redir" in path:
            return True

        return False

    @staticmethod
    def _audio_mime(ext: str) -> str:
        return {
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
        }.get(ext, "audio/mp3")

    @staticmethod
    def _image_mime(ext: str) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".svg": "image/svg+xml",
        }.get(ext, "image/jpeg")

    def _file_to_data_uri(self, path: str, mime: str) -> str:
        # Small, local-only work; no network. Guard against giant files.
        size = os.path.getsize(path)
        if size > self.MAX_LOCAL_BYTES:
            raise ValueError(
                f"Local file too large ({size} bytes). Max allowed is {self.MAX_LOCAL_BYTES}."
            )
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            raise ValueError(f"Failed to read local file: {e}")

    # An optional lightweight validator so callers get a nice error earlier if input is not a string.
    @field_validator("input")
    @classmethod
    def _validate_input_is_str(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("Protect input must be a string")
        return v
