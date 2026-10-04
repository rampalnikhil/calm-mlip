"""Current filesystem artifact-store implementation.

Artifacts are written under::

    <root>/runs/<run_id_short>/<category>/<filename>

Different content for an existing filename is stored in a sibling
``.versions/<sha256>/<filename>`` directory. Published files are never replaced.

The authoritative database stores canonical absolute ``file://`` URIs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


class FSArtifactStore:
    """Persist run-scoped artifact bytes beneath one filesystem root."""

    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _runs_dir(self) -> Path:
        return self._root / "runs"

    def _run_dir(self, run_id_short: str) -> Path:
        run = self._relative_path(run_id_short)
        if len(run.parts) != 1:
            raise ValueError("Artifact run IDs must be a single path component.")
        return self._checked_path(self._runs_dir() / run)

    @staticmethod
    def _relative_path(value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("Artifact paths must be nonempty relative paths without '..'.")
        return path

    def _checked_path(self, path: Path) -> Path:
        """Reject traversal and existing symlinks, including dangling links."""
        relative = path.relative_to(self._root)
        current = self._root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Artifact paths must not contain symbolic links.")
        if not path.resolve().is_relative_to(self._root):
            raise ValueError("Artifact path is outside the artifact root.")
        return path

    def _mkdir(self, path: Path) -> None:
        self._checked_path(path).mkdir(parents=True, exist_ok=True)
        self._checked_path(path)

    def ensure_run_layout(self, *, run_id_short: str) -> None:
        run_dir = self._run_dir(run_id_short)
        for category in (
            "logs",
            "plots",
            "structures",
            "data",
            "tables",
            "files",
        ):
            self._mkdir(run_dir / category)

    def _publish(self, path: Path, data: bytes) -> bool:
        """Atomically publish complete bytes, or compare an existing file."""
        self._mkdir(path.parent)
        self._checked_path(path)
        if path.exists():
            return path.read_bytes() == data
        # A hard link publishes a complete file exclusively. Unlike replace(),
        # concurrent writers cannot overwrite a path already returned to a caller.
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            self._checked_path(path)
            try:
                os.link(temporary_path, path)
                return True
            except FileExistsError:
                return self._checked_path(path).read_bytes() == data
        finally:
            temporary_path.unlink(missing_ok=True)

    def put_bytes(
        self,
        *,
        run_id_short: str,
        category: str,
        filename: str,
        data: bytes,
    ) -> Path:
        """Publish immutable bytes and return their absolute path."""

        # Validate all user components before creating any run directories.
        category_path = self._relative_path(category)
        filename_path = self._relative_path(filename)
        path = self._checked_path(
            self._run_dir(run_id_short) / category_path / filename_path
        )
        self.ensure_run_layout(run_id_short=run_id_short)
        if self._publish(path, data):
            return path
        digest = hashlib.sha256(data).hexdigest()
        version = path.parent / ".versions" / digest / path.name
        if not self._publish(version, data):
            raise ValueError("Artifact version path contains unexpected content.")
        return version

    @staticmethod
    def uri_for(path: Path) -> str:
        """Return the canonical absolute local-file URI for ``path``."""

        if not isinstance(path, Path):
            raise TypeError("Artifact paths must be pathlib.Path instances.")
        if not path.is_absolute():
            raise ValueError("Artifact paths must be absolute.")
        return path.resolve().as_uri()
