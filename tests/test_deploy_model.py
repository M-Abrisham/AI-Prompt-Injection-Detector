"""Tests for scripts/deploy_model.py.

Covers normal deploy, unchanged-file skip, timestamped + plain backup
creation, backup verification, rollback from .bak, missing source file
exit code, and KNOWN_HASHES regex replacement -- all without touching
the real model directories.
"""

import hashlib
import os
import re
import sys
import tempfile
import unittest
from unittest import mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_file(path, content=b"fake model bytes"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


_INIT_TEMPLATE = """\
# stub __init__.py
KNOWN_HASHES = {
    "model.pkl": "aaaa",
    "tfidf_vectorizer.pkl": "bbbb",
}
"""


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class TestImport(unittest.TestCase):
    def test_module_imports(self):
        import scripts.deploy_model as mod
        self.assertTrue(hasattr(mod, "deploy"))
        self.assertTrue(hasattr(mod, "rollback"))
        self.assertTrue(hasattr(mod, "_sha256"))
        self.assertTrue(hasattr(mod, "_backup_file"))
        self.assertTrue(hasattr(mod, "_build_parser"))

    def test_parser_rollback_flag(self):
        from scripts.deploy_model import _build_parser
        parser = _build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.rollback)
        args_rb = parser.parse_args(["--rollback"])
        self.assertTrue(args_rb.rollback)


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

class TestSha256(unittest.TestCase):
    def test_known_digest(self):
        from scripts.deploy_model import _sha256 as sha
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            expected = hashlib.sha256(b"hello").hexdigest()
            self.assertEqual(sha(path), expected)
        finally:
            os.unlink(path)

    def test_different_content_different_digest(self):
        from scripts.deploy_model import _sha256 as sha
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"aaa")
            p1 = f.name
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"bbb")
            p2 = f.name
        try:
            self.assertNotEqual(sha(p1), sha(p2))
        finally:
            os.unlink(p1)
            os.unlink(p2)


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

class TestBackupFile(unittest.TestCase):
    def _make_dst(self, tmpdir, name="model.pkl", content=b"data"):
        path = os.path.join(tmpdir, name)
        _write_file(path, content)
        return path

    def test_creates_plain_bak(self):
        from scripts.deploy_model import _backup_file
        with tempfile.TemporaryDirectory() as td:
            dst = self._make_dst(td)
            _backup_file(dst)
            self.assertTrue(os.path.exists(dst + ".bak"))

    def test_creates_timestamped_bak(self):
        from scripts.deploy_model import _backup_file
        with tempfile.TemporaryDirectory() as td:
            dst = self._make_dst(td)
            ts_bak = _backup_file(dst)
            # Timestamped backup path returned
            self.assertTrue(os.path.exists(ts_bak))
            # Name matches expected pattern
            self.assertRegex(
                os.path.basename(ts_bak),
                r"model\.pkl\.\d{4}-\d{2}-\d{2}-\d{6}\.bak",
            )

    def test_backup_size_matches_original(self):
        from scripts.deploy_model import _backup_file
        content = b"x" * 1024
        with tempfile.TemporaryDirectory() as td:
            dst = self._make_dst(td, content=content)
            ts_bak = _backup_file(dst)
            self.assertEqual(os.path.getsize(ts_bak), len(content))
            self.assertEqual(os.path.getsize(dst + ".bak"), len(content))

    def test_backup_exits_1_when_source_unreadable(self):
        """If dst is missing, _backup_file must sys.exit(1)."""
        from scripts.deploy_model import _backup_file
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "ghost.pkl")
            with self.assertRaises(SystemExit) as ctx:
                _backup_file(missing)
            self.assertEqual(ctx.exception.code, 1)


# ---------------------------------------------------------------------------
# Normal deploy
# ---------------------------------------------------------------------------

class TestDeploy(unittest.TestCase):
    def _setup_dirs(self, td, src_content=b"model v2", dst_content=b"model v1"):
        src_dir = os.path.join(td, "processed")
        dst_dir = os.path.join(td, "models")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        init_path = os.path.join(dst_dir, "__init__.py")
        _write_file(init_path, _INIT_TEMPLATE.encode())
        return src_dir, dst_dir, init_path

    def test_copies_files_to_dest(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"new model " + fname.encode())

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 0)

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                self.assertTrue(os.path.exists(os.path.join(dst_dir, fname)))

    def test_hashes_updated_in_init(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"payload_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()

            # Old placeholder hashes must be gone
            self.assertNotIn('"aaaa"', content)
            self.assertNotIn('"bbbb"', content)
            # Real 64-char hex digests must be present
            self.assertRegex(content, r'"model\.pkl":\s*"[0-9a-f]{64}"')
            self.assertRegex(content, r'"tfidf_vectorizer\.pkl":\s*"[0-9a-f]{64}"')

    def test_backup_created_when_dest_exists(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"v2_" + fname.encode())
                _write_file(os.path.join(dst_dir, fname), b"v1_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            # Plain .bak files should exist for both model files
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                self.assertTrue(
                    os.path.exists(os.path.join(dst_dir, fname + ".bak")),
                    f"Expected .bak for {fname}",
                )

    def test_timestamped_backup_created_when_dest_exists(self):
        from scripts.deploy_model import deploy
        import glob
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"v2_" + fname.encode())
                _write_file(os.path.join(dst_dir, fname), b"v1_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                ts_baks = glob.glob(
                    os.path.join(dst_dir, f"{fname}.[0-9][0-9][0-9][0-9]-*.bak")
                )
                self.assertTrue(
                    len(ts_baks) >= 1,
                    f"Expected at least one timestamped .bak for {fname}",
                )

    def test_exit_code_0_on_success(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 0)


# ---------------------------------------------------------------------------
# Skip when unchanged
# ---------------------------------------------------------------------------

class TestDeployUnchanged(unittest.TestCase):
    def test_skips_copy_when_hash_identical(self):
        """When src and dst have the same SHA-256, shutil.copy2 must not run."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            init_path = os.path.join(dst_dir, "__init__.py")
            _write_file(init_path, _INIT_TEMPLATE.encode())

            identical_content = b"identical bytes"
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), identical_content)
                _write_file(os.path.join(dst_dir, fname), identical_content)

            # Patch shutil.copy2 to detect unexpected calls
            copy_calls = []
            original_copy2 = __import__("shutil").copy2

            def tracked_copy2(src, dst):
                # Only track copies of model files, not backups or init
                if dst.endswith((".pkl",)):
                    copy_calls.append(dst)
                return original_copy2(src, dst)

            with mock.patch("shutil.copy2", side_effect=tracked_copy2):
                with self.assertRaises(SystemExit) as ctx:
                    deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            self.assertEqual(ctx.exception.code, 0)
            # No .pkl files should have been overwritten
            self.assertEqual(copy_calls, [], "copy2 should not have been called for unchanged files")

    def test_unchanged_file_preserves_mtime(self):
        """The destination mtime must not change when skipping an identical file."""
        from scripts.deploy_model import deploy
        import time
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            init_path = os.path.join(dst_dir, "__init__.py")
            _write_file(init_path, _INIT_TEMPLATE.encode())

            content = b"same content"
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), content)
                _write_file(os.path.join(dst_dir, fname), content)

            dst_mtimes_before = {
                fname: os.path.getmtime(os.path.join(dst_dir, fname))
                for fname in ["model.pkl", "tfidf_vectorizer.pkl"]
            }

            # Small sleep so any copy would produce a measurably different mtime
            time.sleep(0.05)

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                mtime_after = os.path.getmtime(os.path.join(dst_dir, fname))
                self.assertAlmostEqual(
                    dst_mtimes_before[fname],
                    mtime_after,
                    delta=0.01,
                    msg=f"mtime changed for {fname} even though content was identical",
                )


# ---------------------------------------------------------------------------
# Missing source file
# ---------------------------------------------------------------------------

class TestDeployMissingSource(unittest.TestCase):
    def test_exit_1_when_source_missing(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            init_path = os.path.join(dst_dir, "__init__.py")
            _write_file(init_path, _INIT_TEMPLATE.encode())
            # Do NOT create any source .pkl files

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 1)

    def test_exit_1_when_only_one_source_missing(self):
        """Even if one of the two model files is missing, exit code must be 1."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            init_path = os.path.join(dst_dir, "__init__.py")
            _write_file(init_path, _INIT_TEMPLATE.encode())
            # Only create the first file
            _write_file(os.path.join(src_dir, "model.pkl"), b"data")

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 1)


# ---------------------------------------------------------------------------
# KNOWN_HASHES regex replacement
# ---------------------------------------------------------------------------

class TestKnownHashesReplacement(unittest.TestCase):
    def test_replaces_existing_block(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            init_path = os.path.join(dst_dir, "__init__.py")
            _write_file(init_path, _INIT_TEMPLATE.encode())

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"unique_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                result = f.read()

            # Exactly one KNOWN_HASHES block must remain
            matches = re.findall(r"KNOWN_HASHES\s*=\s*\{", result)
            self.assertEqual(len(matches), 1)

    def test_does_not_touch_init_when_no_block_present(self):
        """If KNOWN_HASHES is absent, deploy() warns but must not crash."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            init_path = os.path.join(dst_dir, "__init__.py")
            # Write an init with no KNOWN_HASHES
            _write_file(init_path, b"# no hashes here\n")

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data")

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            # Should still exit 0 (warning only, not an error)
            self.assertEqual(ctx.exception.code, 0)

    def test_hash_values_are_64_char_hex(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            init_path = os.path.join(dst_dir, "__init__.py")
            _write_file(init_path, _INIT_TEMPLATE.encode())

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), fname.encode() * 10)

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()

            for digest_match in re.findall(r'"([0-9a-f]+)"', content):
                if len(digest_match) == 64:
                    # Confirm it is valid hex
                    int(digest_match, 16)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

class TestRollback(unittest.TestCase):
    def _prepare_with_baks(self, td, bak_content=b"old model"):
        dst_dir = os.path.join(td, "models")
        os.makedirs(dst_dir)
        for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
            live = os.path.join(dst_dir, fname)
            bak = live + ".bak"
            _write_file(live, b"new model")
            _write_file(bak, bak_content)
        return dst_dir

    def test_rollback_restores_bak_files(self):
        from scripts.deploy_model import rollback
        bak_content = b"restored old model"
        with tempfile.TemporaryDirectory() as td:
            dst_dir = self._prepare_with_baks(td, bak_content)

            with self.assertRaises(SystemExit) as ctx:
                rollback(dest_dir=dst_dir)
            self.assertEqual(ctx.exception.code, 0)

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                with open(os.path.join(dst_dir, fname), "rb") as f:
                    self.assertEqual(f.read(), bak_content)

    def test_rollback_exit_1_when_bak_missing(self):
        from scripts.deploy_model import rollback
        with tempfile.TemporaryDirectory() as td:
            dst_dir = os.path.join(td, "models")
            os.makedirs(dst_dir)
            # Create live files but NO .bak files

            with self.assertRaises(SystemExit) as ctx:
                rollback(dest_dir=dst_dir)
            self.assertEqual(ctx.exception.code, 1)

    def test_rollback_exit_1_when_only_one_bak_missing(self):
        from scripts.deploy_model import rollback
        with tempfile.TemporaryDirectory() as td:
            dst_dir = os.path.join(td, "models")
            os.makedirs(dst_dir)
            # Create .bak for model.pkl only
            _write_file(os.path.join(dst_dir, "model.pkl"), b"live")
            _write_file(os.path.join(dst_dir, "model.pkl.bak"), b"old")
            # tfidf_vectorizer.pkl.bak is absent

            with self.assertRaises(SystemExit) as ctx:
                rollback(dest_dir=dst_dir)
            self.assertEqual(ctx.exception.code, 1)

    def test_rollback_exit_0_when_all_baks_present(self):
        from scripts.deploy_model import rollback
        with tempfile.TemporaryDirectory() as td:
            dst_dir = self._prepare_with_baks(td)
            with self.assertRaises(SystemExit) as ctx:
                rollback(dest_dir=dst_dir)
            self.assertEqual(ctx.exception.code, 0)


# ---------------------------------------------------------------------------
# Char-level TF-IDF vectorizer (conditionally required)
# ---------------------------------------------------------------------------

class TestCharVectorizerRequired(unittest.TestCase):
    """char_tfidf_vectorizer.pkl must be deployed when it exists alongside model.pkl."""

    def _setup_dirs(self, td):
        src_dir = os.path.join(td, "processed")
        dst_dir = os.path.join(td, "models")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        init_path = os.path.join(dst_dir, "__init__.py")
        _write_file(init_path, _INIT_TEMPLATE.encode())
        return src_dir, dst_dir, init_path

    def test_char_vectorizer_copied_when_present(self):
        """If char_tfidf_vectorizer.pkl exists in source, it must be copied."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl", "char_tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 0)
            self.assertTrue(
                os.path.exists(os.path.join(dst_dir, "char_tfidf_vectorizer.pkl")),
                "char_tfidf_vectorizer.pkl must be deployed when present in source",
            )

    def test_char_vectorizer_hash_in_init(self):
        """char_tfidf_vectorizer.pkl must appear in KNOWN_HASHES after deploy."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl", "char_tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"payload_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()
            self.assertRegex(
                content,
                r'"char_tfidf_vectorizer\.pkl":\s*"[0-9a-f]{64}"',
                "char_tfidf_vectorizer.pkl must have a SHA-256 entry in KNOWN_HASHES",
            )

    def test_deploy_succeeds_without_char_vectorizer(self):
        """When char_tfidf_vectorizer.pkl is absent, deploy must still work."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 0)

    def test_char_vectorizer_not_in_hash_when_absent(self):
        """When char_tfidf_vectorizer.pkl is absent, it must NOT appear in KNOWN_HASHES."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()
            self.assertNotIn("char_tfidf_vectorizer.pkl", content)

    def test_rollback_restores_char_vectorizer_bak(self):
        """Rollback must restore char_tfidf_vectorizer.pkl if its .bak exists."""
        from scripts.deploy_model import rollback
        bak_content = b"old char vectorizer"
        with tempfile.TemporaryDirectory() as td:
            dst_dir = os.path.join(td, "models")
            os.makedirs(dst_dir)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl", "char_tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(dst_dir, fname), b"new")
                _write_file(os.path.join(dst_dir, fname + ".bak"), bak_content)

            with self.assertRaises(SystemExit) as ctx:
                rollback(dest_dir=dst_dir)
            self.assertEqual(ctx.exception.code, 0)

            with open(os.path.join(dst_dir, "char_tfidf_vectorizer.pkl"), "rb") as f:
                self.assertEqual(f.read(), bak_content)


# ---------------------------------------------------------------------------
# Sidecar regeneration (F-AR6)
# ---------------------------------------------------------------------------

def _parse_sidecar(raw):
    """Local mirror of safe_pickle._parse_sidecar so this test has no hard
    dependency on the (heavy) na0s import for the deploy-path assertions."""
    raw = raw.strip()
    if raw.startswith("v1:"):
        parts = raw.split(":", 2)
        if len(parts) == 3:
            return parts[2]
    return raw


class TestSidecarRegen(unittest.TestCase):
    """F-AR6: deploy_model must (re)write each model file's .sha256 sidecar from
    the freshly-deployed bytes — deploy copies the .pkl but not its sidecar, so
    without this the destination keeps a stale sidecar that no longer matches the
    .pkl it guards."""

    def _setup_dirs(self, td):
        src_dir = os.path.join(td, "processed")
        dst_dir = os.path.join(td, "models")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        init_path = os.path.join(dst_dir, "__init__.py")
        _write_file(init_path, _INIT_TEMPLATE.encode())
        return src_dir, dst_dir, init_path

    def test_sidecar_written_and_matches_pkl(self):
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"fresh_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                pkl = os.path.join(dst_dir, fname)
                sidecar = pkl + ".sha256"
                self.assertTrue(os.path.exists(sidecar), f"sidecar missing for {fname}")
                parsed = _parse_sidecar(open(sidecar).read())
                self.assertEqual(parsed, _sha256(pkl),
                                 f"sidecar for {fname} does not match its .pkl")

    def test_unchanged_branch_refreshes_stale_sidecar(self):
        """The bug exactly: .pkl unchanged but the shipped sidecar is stale.
        deploy's unchanged-skip path must still rewrite the sidecar."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                content = b"identical_" + fname.encode()
                _write_file(os.path.join(src_dir, fname), content)
                _write_file(os.path.join(dst_dir, fname), content)  # same bytes -> skip copy
                # Seed a STALE sidecar that does NOT match the file.
                _write_file(os.path.join(dst_dir, fname + ".sha256"), b"v1:sha256:" + b"0" * 64)

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                pkl = os.path.join(dst_dir, fname)
                parsed = _parse_sidecar(open(pkl + ".sha256").read())
                self.assertEqual(parsed, _sha256(pkl),
                                 f"stale sidecar for {fname} was not refreshed")


class TestShippedSidecarsFresh(unittest.TestCase):
    """Regression guard against the real shipped sidecars going stale again
    (the original F-AR6 finding: model.pkl.sha256 / tfidf_vectorizer.pkl.sha256
    did not match their .pkl)."""

    def test_shipped_sidecars_match_their_pkl(self):
        from scripts.deploy_model import DEST_DIR
        checked = 0
        for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
            pkl = os.path.join(DEST_DIR, fname)
            sidecar = pkl + ".sha256"
            if not (os.path.exists(pkl) and os.path.exists(sidecar)):
                continue
            parsed = _parse_sidecar(open(sidecar).read())
            self.assertEqual(
                parsed, _sha256(pkl),
                f"shipped {fname}.sha256 is stale — does not match {fname}",
            )
            checked += 1
        if checked == 0:
            self.skipTest("no shipped .pkl + .sha256 pairs found in this build")


# ---------------------------------------------------------------------------
# Failure-mode branch coverage (TC1)
# ---------------------------------------------------------------------------

class TestDeployFailureModes(unittest.TestCase):
    """Exercise the error branches that the happy-path tests never reach:
    copy failure, rollback restore failure, unreadable __init__, sidecar write
    failure, and backup size-mismatch — each must exit 1 (or sys.exit(1))."""

    def _setup_dirs(self, td):
        src_dir = os.path.join(td, "processed")
        dst_dir = os.path.join(td, "models")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        init_path = os.path.join(dst_dir, "__init__.py")
        _write_file(init_path, _INIT_TEMPLATE.encode())
        return src_dir, dst_dir, init_path

    def test_deploy_copy_failure_exits_1(self):
        """shutil.copy2 raising OSError during deploy must exit 1."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())
            with mock.patch("shutil.copy2", side_effect=OSError("disk full")):
                with self.assertRaises(SystemExit) as ctx:
                    deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 1)

    def test_rollback_copy_failure_exits_1(self):
        """shutil.copy2 raising OSError during rollback must exit 1."""
        from scripts.deploy_model import rollback
        with tempfile.TemporaryDirectory() as td:
            dst_dir = os.path.join(td, "models")
            os.makedirs(dst_dir)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(dst_dir, fname), b"live")
                _write_file(os.path.join(dst_dir, fname + ".bak"), b"old")
            with mock.patch("shutil.copy2", side_effect=OSError("read-only fs")):
                with self.assertRaises(SystemExit) as ctx:
                    rollback(dest_dir=dst_dir)
            self.assertEqual(ctx.exception.code, 1)

    def test_deploy_unreadable_init_exits_1(self):
        """An __init__ path that cannot be read (here: a directory) exits 1."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir = os.path.join(td, "processed")
            dst_dir = os.path.join(td, "models")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())
            # init_path is a directory -> open(..., "r") raises OSError.
            init_dir = os.path.join(dst_dir, "init_as_dir")
            os.makedirs(init_dir)
            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_dir)
            self.assertEqual(ctx.exception.code, 1)

    def test_write_sidecar_oserror_exits_1(self):
        """_write_sidecar must exit 1 when the sidecar path is unwritable."""
        from scripts.deploy_model import _write_sidecar
        with tempfile.TemporaryDirectory() as td:
            pkl = os.path.join(td, "model.pkl")
            _write_file(pkl, b"x")
            os.makedirs(pkl + ".sha256")  # directory at the sidecar path
            with self.assertRaises(SystemExit) as ctx:
                _write_sidecar(pkl, "deadbeef")
            self.assertEqual(ctx.exception.code, 1)

    def test_backup_size_mismatch_exits_1(self):
        """_backup_file must exit 1 when the backup size differs from the original."""
        from scripts.deploy_model import _backup_file
        with tempfile.TemporaryDirectory() as td:
            dst = os.path.join(td, "model.pkl")
            _write_file(dst, b"data")
            real_getsize = os.path.getsize

            def fake_getsize(p):
                # Report a wrong size for the timestamped backup only.
                base = os.path.basename(p)
                if base.startswith("model.pkl.") and base.endswith(".bak") and "-" in base:
                    return 999999
                return real_getsize(p)

            with mock.patch("os.path.getsize", side_effect=fake_getsize):
                with self.assertRaises(SystemExit) as ctx:
                    _backup_file(dst)
            self.assertEqual(ctx.exception.code, 1)



# ---------------------------------------------------------------------------
# Preserve-and-merge: bundled pkls not re-emitted this run must survive
# ---------------------------------------------------------------------------

# A 64-hex digest that no real fake-file content will collide with, used to
# prove an entry was PRESERVED verbatim (not recomputed from a copied file).
_EMBED_ORIG = "09" * 32
_SCALER_ORIG = "51" * 32

_INIT_WITH_EMBED = """\
# stub __init__.py
KNOWN_HASHES = {
    "model.pkl": "aaaa",
    "structural_scaler.pkl": "%s",
    "model_embedding.pkl": "%s",
    "tfidf_vectorizer.pkl": "bbbb",
}
""" % (_SCALER_ORIG, _EMBED_ORIG)


class TestKnownHashesPreserveMerge(unittest.TestCase):
    """deploy() must MERGE fresh digests into the existing KNOWN_HASHES,
    preserving entries for bundled pickles it does not re-emit this run.

    These tests demonstrate the M4(b) drop bug: each one FAILS on the old
    rebuild-from-scratch code (which seeded new_hashes={} and re-wrote the
    whole dict from only copied files) and PASSES after the preserve-merge fix.
    """

    def _setup_dirs(self, td, init_template):
        src_dir = os.path.join(td, "processed")
        dst_dir = os.path.join(td, "models")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        init_path = os.path.join(dst_dir, "__init__.py")
        _write_file(init_path, init_template.encode())
        return src_dir, dst_dir, init_path

    def test_model_embedding_preserved_when_absent_from_source(self):
        """T1 (headline): a TF-IDF-only redeploy must NOT drop model_embedding.pkl.

        RED on old code (entry erased), GREEN after preserve-merge.
        """
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_WITH_EMBED)
            # Only model.pkl + tfidf in source: a normal redeploy that does
            # NOT re-emit the embedding model.
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"new_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()
            # The original embedding digest must survive UNCHANGED.
            self.assertIn(
                '"model_embedding.pkl": "%s"' % _EMBED_ORIG,
                content,
                "model_embedding.pkl entry was dropped from KNOWN_HASHES",
            )

    def test_structural_scaler_preserved_when_absent_from_source(self):
        """T2 (G2): structural_scaler.pkl entry must survive when not in source."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_WITH_EMBED)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"new_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()
            self.assertIn(
                '"structural_scaler.pkl": "%s"' % _SCALER_ORIG,
                content,
                "structural_scaler.pkl entry was dropped from KNOWN_HASHES",
            )

    def test_copied_file_hash_is_updated_not_frozen(self):
        """T3: a copied file's stale hash must be REPLACED with the real SHA-256,
        proving merge updates (not freezes) entries for files it touches."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_WITH_EMBED)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"real_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()
            # Stale placeholder hashes for copied files must be gone.
            self.assertNotIn('"aaaa"', content)
            self.assertNotIn('"bbbb"', content)
            # model.pkl entry must equal the real SHA-256 of the deployed file.
            real_model_digest = _sha256(os.path.join(dst_dir, "model.pkl"))
            self.assertIn(
                '"model.pkl": "%s"' % real_model_digest,
                content,
            )

    def test_idempotent_second_run(self):
        """T3b (G4): two consecutive deploys with identical inputs must leave
        KNOWN_HASHES byte-identical (no spurious churn)."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_WITH_EMBED)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"stable_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            with open(init_path) as f:
                after_first = f.read()

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            with open(init_path) as f:
                after_second = f.read()

            self.assertEqual(after_first, after_second)
            # And the preserved embedding entry is still intact.
            self.assertIn(
                '"model_embedding.pkl": "%s"' % _EMBED_ORIG, after_second
            )

    def test_char_vectorizer_still_absent_after_preserve_merge(self):
        """Preserve-merge must NOT invent char_tfidf_vectorizer.pkl: when it is
        absent from both source and the existing dict, it stays absent."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_WITH_EMBED)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"new_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                content = f.read()
            self.assertNotIn("char_tfidf_vectorizer.pkl", content)


# ---------------------------------------------------------------------------
# Reader brace-safety (item #01 hardening): a nested '}' in a KNOWN_HASHES
# value must NOT make _parse_known_hashes return {} and silently drop bundled
# entries through deploy(). The old non-greedy regex r"\{.*?\}" truncated at
# the first inner '}', literal_eval failed, the reader returned {}, and
# preserve-and-merge erased model_embedding.pkl / structural_scaler.pkl. The
# AST reader slices the assignment's value node, so inner braces are inert.
# ---------------------------------------------------------------------------

# Like _INIT_WITH_EMBED, but one entry carries an inner '}' (a comment with a
# brace) — the exact shape that truncated the old regex reader. The bundled
# embedding + scaler digests below MUST survive a redeploy that does not
# re-emit them.
_INIT_WITH_EMBED_NESTED_BRACE = """\
# stub __init__.py
KNOWN_HASHES = {
    "model.pkl": "aaaa",  # layout note: {a:b}
    "structural_scaler.pkl": "%s",
    "model_embedding.pkl": "%s",
    "tfidf_vectorizer.pkl": "bbbb",
}
""" % (_SCALER_ORIG, _EMBED_ORIG)


class TestKnownHashesReaderNestedBrace(unittest.TestCase):
    """item #01 hardening: the READER must parse a KNOWN_HASHES block whose
    value contains a nested '}' and PRESERVE its not-re-emitted bundled entries
    through deploy(). RED on the old regex reader (literal_eval fails -> {} ->
    entries dropped), GREEN on the AST-node reader."""

    def _setup_dirs(self, td, init_template):
        src_dir = os.path.join(td, "processed")
        dst_dir = os.path.join(td, "models")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        init_path = os.path.join(dst_dir, "__init__.py")
        _write_file(init_path, init_template.encode())
        return src_dir, dst_dir, init_path

    def test_parse_known_hashes_unit_survives_nested_brace(self):
        """Unit: _parse_known_hashes returns ALL four entries (not {}) for a
        block with a nested '}'. The old regex returned {} here."""
        from scripts.deploy_model import _parse_known_hashes
        parsed = _parse_known_hashes(_INIT_WITH_EMBED_NESTED_BRACE)
        self.assertEqual(
            set(parsed.keys()),
            {"model.pkl", "structural_scaler.pkl",
             "model_embedding.pkl", "tfidf_vectorizer.pkl"},
            "nested '}' in a value truncated the reader -> entries lost",
        )
        # The preserved bundled digests are read verbatim.
        self.assertEqual(parsed["model_embedding.pkl"], _EMBED_ORIG)
        self.assertEqual(parsed["structural_scaler.pkl"], _SCALER_ORIG)

    def test_bundled_entries_preserved_through_deploy_with_nested_brace(self):
        """End-to-end: a TF-IDF-only redeploy over a nested-brace block must
        NOT drop model_embedding.pkl / structural_scaler.pkl. This is the bug
        item #01 was meant to fix, made latent-proof by the AST reader."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(
                td, _INIT_WITH_EMBED_NESTED_BRACE
            )
            # Only model.pkl + tfidf in source: the embedding/scaler are NOT
            # re-emitted this run, so they can only survive via the reader.
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"new_" + fname.encode())

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 0)

            with open(init_path) as f:
                content = f.read()
            # Both preserved bundled digests survive UNCHANGED.
            self.assertIn(
                '"model_embedding.pkl": "%s"' % _EMBED_ORIG, content,
                "model_embedding.pkl dropped: nested '}' broke the reader",
            )
            self.assertIn(
                '"structural_scaler.pkl": "%s"' % _SCALER_ORIG, content,
                "structural_scaler.pkl dropped: nested '}' broke the reader",
            )
            # Result is valid Python and round-trips (file is never corrupt).
            import ast
            from scripts.deploy_model import _find_known_hashes_assign
            node = _find_known_hashes_assign(ast.parse(content))
            self.assertIsNotNone(node)
            self.assertEqual(
                set(ast.literal_eval(node.value).keys()),
                {"model.pkl", "structural_scaler.pkl",
                 "model_embedding.pkl", "tfidf_vectorizer.pkl"},
            )


# ---------------------------------------------------------------------------
# AST-based KNOWN_HASHES rewrite (item #14): brace-safe + parse-verified
# ---------------------------------------------------------------------------

# An __init__ whose KNOWN_HASHES block carries an inner '}' (in a comment on
# an entry line). The OLD brace-fragile re.sub r"KNOWN_HASHES\s*=\s*\{[^}]*\}"
# truncates at that first inner '}' and emits invalid Python
# (proven: "SyntaxError: unexpected indent"). The AST slice-rewrite splices the
# whole assignment span, so the inner brace is irrelevant.
_INIT_NESTED_BRACE = """\
# stub __init__.py
KNOWN_HASHES = {
    "model.pkl": "aaaa",  # layout: {a:b}
    "tfidf_vectorizer.pkl": "bbbb",
}

SOMETHING_ELSE = 1
"""


class TestKnownHashesAstRewrite(unittest.TestCase):
    """Item #14: the KNOWN_HASHES rewrite is AST-based, not a brace-fragile
    regex, and the result is ast.parse + ast.literal_eval verified before any
    write (fail-closed)."""

    def _setup_dirs(self, td, init_template):
        src_dir = os.path.join(td, "processed")
        dst_dir = os.path.join(td, "models")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        init_path = os.path.join(dst_dir, "__init__.py")
        _write_file(init_path, init_template.encode())
        return src_dir, dst_dir, init_path

    def test_nested_brace_value_survives_and_parses(self):
        """T1 (headline, red on old re.sub / green on AST): a KNOWN_HASHES
        block containing an inner '}' must rewrite to VALID Python that
        round-trips via ast.literal_eval, and code after the block survives."""
        import ast
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_NESTED_BRACE)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"uniq_" + fname.encode())

            with self.assertRaises(SystemExit) as ctx:
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 0)

            with open(init_path) as f:
                result = f.read()

            # Must be valid Python (the old re.sub produced SyntaxError here).
            tree = ast.parse(result)  # raises if invalid -> test fails
            # KNOWN_HASHES round-trips to the two real 64-hex digests.
            from scripts.deploy_model import _find_known_hashes_assign
            node = _find_known_hashes_assign(tree)
            self.assertIsNotNone(node)
            parsed = ast.literal_eval(node.value)
            self.assertEqual(set(parsed.keys()), {"model.pkl", "tfidf_vectorizer.pkl"})
            for digest in parsed.values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
            # The stale placeholder values are gone and there is exactly one block.
            self.assertNotIn('"aaaa"', result)
            self.assertNotIn('"bbbb"', result)
            self.assertEqual(len(re.findall(r"KNOWN_HASHES\s*=\s*\{", result)), 1)
            # Code AFTER the block is preserved (proves the splice ended at the
            # dict's closing brace, not the comment's inner '}').
            self.assertIn("SOMETHING_ELSE = 1", result)

    def test_rewrite_helper_round_trips_nested_brace(self):
        """T1 (unit): _rewrite_known_hashes on a nested-brace block returns
        'ok' and the spliced text ast-parses + literal_evals to new_hashes."""
        import ast
        from scripts.deploy_model import _rewrite_known_hashes, _find_known_hashes_assign
        new = {"model.pkl": "1" * 64, "tfidf_vectorizer.pkl": "2" * 64}
        updated, status = _rewrite_known_hashes(_INIT_NESTED_BRACE, new)
        self.assertEqual(status, "ok")
        node = _find_known_hashes_assign(ast.parse(updated))
        self.assertEqual(ast.literal_eval(node.value), new)

    def test_malformed_rewrite_rejected_pre_write(self):
        """T2 (fail-closed, G3): if the rendered literal would be invalid
        Python, deploy() exits 1 and the live __init__.py is byte-identical
        (the corrupt rewrite is NEVER written)."""
        from scripts import deploy_model
        from scripts.deploy_model import deploy

        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_TEMPLATE)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())

            with open(init_path) as f:
                before = f.read()

            # Force the renderer to emit a literal that does NOT parse, so the
            # output ast.parse verify trips and the rewrite is refused.
            def broken_render(_hashes):
                return 'KNOWN_HASHES = {"oops": }'  # invalid Python

            with mock.patch.object(
                deploy_model, "_render_known_hashes", side_effect=broken_render
            ):
                with self.assertRaises(SystemExit) as ctx:
                    deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 1)

            with open(init_path) as f:
                after = f.read()
            # The corrupt rewrite was never written: file unchanged.
            self.assertEqual(before, after)

    def test_round_trip_mismatch_rejected(self):
        """T2b: a syntactically-valid rewrite whose dict does NOT equal the
        intended new_hashes is rejected (round-trip guard), file unchanged."""
        from scripts import deploy_model
        from scripts.deploy_model import deploy

        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_TEMPLATE)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())

            with open(init_path) as f:
                before = f.read()

            # Valid Python, but WRONG contents (does not match new_hashes).
            def wrong_render(_hashes):
                return 'KNOWN_HASHES = {\n    "evil.pkl": "deadbeef",\n}'

            with mock.patch.object(
                deploy_model, "_render_known_hashes", side_effect=wrong_render
            ):
                with self.assertRaises(SystemExit) as ctx:
                    deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            self.assertEqual(ctx.exception.code, 1)

            with open(init_path) as f:
                after = f.read()
            self.assertEqual(before, after)
            self.assertNotIn("evil.pkl", after)

    def test_init_backup_created_and_equals_live_on_failure(self):
        """T3 (G4): on the fail-closed path a __init__.py.bak exists and equals
        the (unchanged) live file."""
        from scripts import deploy_model
        from scripts.deploy_model import deploy

        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_TEMPLATE)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"data_" + fname.encode())

            def broken_render(_hashes):
                return 'KNOWN_HASHES = {"oops": }'

            with mock.patch.object(
                deploy_model, "_render_known_hashes", side_effect=broken_render
            ):
                with self.assertRaises(SystemExit):
                    deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            bak = init_path + ".bak"
            self.assertTrue(os.path.exists(bak), "expected __init__.py.bak on failure")
            with open(init_path) as f:
                live = f.read()
            with open(bak) as f:
                backed = f.read()
            self.assertEqual(live, backed)

    def test_init_backup_created_on_success(self):
        """T3b: a successful KNOWN_HASHES update backs up __init__.py first."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_TEMPLATE)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"fresh_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            self.assertTrue(
                os.path.exists(init_path + ".bak"),
                "expected __init__.py.bak after a real KNOWN_HASHES update",
            )

    def test_idempotent_no_backup_when_unchanged(self):
        """T4 (G5): a second identical deploy leaves __init__.py byte-identical
        (the 'unchanged' path holds under the AST rewrite)."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_TEMPLATE)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"stable_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            with open(init_path) as f:
                after_first = f.read()

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            with open(init_path) as f:
                after_second = f.read()

            self.assertEqual(after_first, after_second)

    def test_rewritten_module_imports(self):
        """T5 (use-case, end-to-end): after deploy() over a nested-brace init,
        the rewritten module compiles AND exposes the expected KNOWN_HASHES
        dict — proving the producer never emits a module that breaks
        `import na0s.models` (predict.py:84 import)."""
        from scripts.deploy_model import deploy
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td, _INIT_NESTED_BRACE)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"mod_" + fname.encode())

            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)

            with open(init_path) as f:
                src = f.read()
            # Execute the rewritten module in an isolated namespace: this is
            # what `import na0s.models` does at load time.
            ns = {}
            code = compile(src, init_path, "exec")
            exec(code, ns)  # noqa: S102 - executing the rewritten module under test
            self.assertIn("KNOWN_HASHES", ns)
            self.assertEqual(
                set(ns["KNOWN_HASHES"].keys()),
                {"model.pkl", "tfidf_vectorizer.pkl"},
            )
            # And the trailing statement executed too.
            self.assertEqual(ns.get("SOMETHING_ELSE"), 1)

    def test_real_init_shape_guard_read_only(self):
        """T7: the helper locates the KNOWN_HASHES node in the REAL
        src/na0s/models/__init__.py and it literal_evals to the live entries.
        Read-only — never writes the real file."""
        import ast
        from scripts.deploy_model import INIT_PATH, _find_known_hashes_assign

        with open(INIT_PATH, encoding="utf-8") as f:
            real = f.read()
        node = _find_known_hashes_assign(ast.parse(real))
        self.assertIsNotNone(node, "KNOWN_HASHES not found in real __init__.py")
        live = ast.literal_eval(node.value)
        self.assertEqual(
            set(live.keys()),
            {
                "model.pkl",
                "structural_scaler.pkl",
                "model_embedding.pkl",
                "tfidf_vectorizer.pkl",
            },
        )
        for digest in live.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_absent_block_returns_absent(self):
        """T5b (unit): the AST walk returns 'absent' (no StopIteration/KeyError)
        when there is no KNOWN_HASHES assignment, so deploy() can warn+exit 0."""
        from scripts.deploy_model import _rewrite_known_hashes
        content = "# nothing here\nX = 1\n"
        out, status = _rewrite_known_hashes(content, {"model.pkl": "z" * 64})
        self.assertEqual(status, "absent")
        self.assertEqual(out, content)


# ---------------------------------------------------------------------------
# Product invariant: every bundled *.pkl is a key in KNOWN_HASHES
# ---------------------------------------------------------------------------

class TestKnownHashesCoverageInvariant(unittest.TestCase):
    """Durable invariant (T4, G3): every *.pkl bundled in src/na0s/models/ MUST
    have an entry in na0s.models.KNOWN_HASHES, since none ship sidecar .sha256
    files — the hardcoded hash is their only integrity source. This guards
    against any future bundled pkl being silently droppable from the dict."""

    def test_every_bundled_pkl_in_known_hashes(self):
        import importlib.resources
        from na0s.models import KNOWN_HASHES

        models_dir = importlib.resources.files("na0s.models")
        bundled = {
            p.name
            for p in models_dir.iterdir()
            if p.name.endswith(".pkl")
        }
        self.assertTrue(bundled, "expected at least one bundled *.pkl")
        missing = bundled - set(KNOWN_HASHES.keys())
        self.assertEqual(
            missing,
            set(),
            "bundled pkl(s) missing from KNOWN_HASHES (no sidecar => "
            "safe_load would raise FileNotFoundError): %s" % sorted(missing),
        )


if __name__ == "__main__":
    unittest.main()
