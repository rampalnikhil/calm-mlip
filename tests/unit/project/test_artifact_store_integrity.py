from concurrent.futures import ThreadPoolExecutor

import pytest

from calm.project.infrastructure.artifacts.fs_store import FSArtifactStore


def _put(store, data, **changes):
    arguments = dict(run_id_short="r_12345678", category="logs", filename="log.txt", data=data)
    arguments.update(changes)
    return store.put_bytes(**arguments)


def test_repeated_filenames_preserve_each_content_version(tmp_path):
    store = FSArtifactStore(tmp_path / "out")
    first = _put(store, b"first")
    second = _put(store, b"second")
    assert first != second
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    assert first.name == second.name == "log.txt"
    reopened = FSArtifactStore(tmp_path / "out")
    assert _put(reopened, b"first") == first
    assert _put(reopened, b"second") == second


def test_concurrent_writes_publish_complete_immutable_files(tmp_path):
    store = FSArtifactStore(tmp_path / "out")
    contents = [bytes([index]) * 10000 for index in range(8)] * 2
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda data: _put(store, data), contents))
    assert [path.read_bytes() for path in paths] == contents
    assert paths[:8] == paths[8:]


def test_rejects_absolute_and_traversing_paths_before_writing(tmp_path):
    root = tmp_path / "out"
    store = FSArtifactStore(root)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"preserve")
    invalid = [
        {"filename": str(outside)}, {"filename": "../../../../outside.txt"},
        {"filename": ""}, {"filename": "."},
        {"category": str(tmp_path)}, {"category": "../escape"},
        {"run_id_short": str(tmp_path)}, {"run_id_short": "../escape"},
    ]
    for changes in invalid:
        with pytest.raises(ValueError):
            _put(store, b"replacement", **changes)
    assert outside.read_bytes() == b"preserve"
    assert list(root.iterdir()) == []


def test_rejects_directory_and_dangling_file_symlinks(tmp_path):
    root = tmp_path / "out"
    store = FSArtifactStore(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        _put(store, b"escape")
    assert list(outside.iterdir()) == []
    (root / "runs").unlink()
    store.ensure_run_layout(run_id_short="r_12345678")
    target = outside / "absent.txt"
    (root / "runs/r_12345678/logs/log.txt").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        _put(store, b"escape")
    assert not target.exists()


def test_safe_relative_subdirectories_remain_supported(tmp_path):
    store = FSArtifactStore(tmp_path / "out")
    path = _put(store, b"nested", category="data/analysis", filename="batch/result.json")
    assert path.read_bytes() == b"nested"
    assert path.is_relative_to(tmp_path / "out")
