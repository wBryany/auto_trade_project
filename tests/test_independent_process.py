"""Exercise the actual runner with isolated sources, never a trading config."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "independent_process.py"


class IndependentProcessTests(unittest.TestCase):
    def run_fixture(self, code):
        with tempfile.TemporaryDirectory(prefix="bot runner spaces ") as directory:
            root = Path(directory)
            package = root / "src" / "btc_futures_bot"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "main.py").write_text(code, encoding="utf-8")
            spec = root / "launch.json"
            spec.write_text(json.dumps({
                "source_root": str(root), "runtime_root": str(root),
                "arguments": ["--config", "path with spaces.json"],
                "stdout": str(root / "out.log"), "stderr": str(root / "err.log"),
            }), encoding="utf-8-sig")
            result = subprocess.run([sys.executable, str(RUNNER), str(spec)],
                                    capture_output=True, timeout=15)
            self.assertFalse(spec.exists())
            return result, (root / "out.log").read_text(encoding="utf-8"), (
                root / "err.log").read_text(encoding="utf-8"), str(root)

    def test_source_arguments_working_directory_and_logs(self):
        result, out, err, root = self.run_fixture(
            "import json, os, sys\n"
            "print(json.dumps([os.getcwd(), sys.argv[1:]]))\n"
            "print('stderr works', file=sys.stderr)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(out), [root, ["--config", "path with spaces.json"]])
        self.assertIn("stderr works", err)

    def test_startup_exception_is_logged_and_fails(self):
        result, out, err, _ = self.run_fixture("raise RuntimeError('fixture failure')\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("RuntimeError: fixture failure", err)


if __name__ == "__main__":
    unittest.main()
