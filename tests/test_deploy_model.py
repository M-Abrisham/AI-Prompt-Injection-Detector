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

    def test_merge_preserves_unretrained_pins(self):
        """F-AR2: deploy MERGES new hashes over the existing KNOWN_HASHES, so a
        pin the retrain does NOT regenerate (model_embedding.pkl, produced only
        by scripts/model_embedding.py) is preserved rather than dropped."""
        from scripts.deploy_model import deploy
        try:
            from na0s.models import KNOWN_HASHES
        except Exception:
            self.skipTest("na0s.models not importable")
        if "model_embedding.pkl" not in KNOWN_HASHES:
            self.skipTest("no model_embedding.pkl pin to preserve in this build")
        with tempfile.TemporaryDirectory() as td:
            src_dir, dst_dir, init_path = self._setup_dirs(td)
            for fname in ["model.pkl", "tfidf_vectorizer.pkl"]:
                _write_file(os.path.join(src_dir, fname), b"cand_" + fname.encode())
            with self.assertRaises(SystemExit):
                deploy(source_dir=src_dir, dest_dir=dst_dir, init_path=init_path)
            content = open(init_path).read()
            # The unretrained embedding pin survives the rewrite (merge, not replace).
            self.assertIn("model_embedding.pkl", content)
            self.assertIn(KNOWN_HASHES["model_embedding.pkl"], content)

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


if __name__ == "__main__":
    unittest.main()
