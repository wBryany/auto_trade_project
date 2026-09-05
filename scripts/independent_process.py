"""WMI-owned dashboard entry point; no Codex runtime or terminal is required."""
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    spec_path.unlink()
    # Redirect before importing the application, so startup failures are logged.
    with open(spec["stdout"], "a", encoding="utf-8", buffering=1) as out, open(
        spec["stderr"], "a", encoding="utf-8", buffering=1
    ) as err:
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        sys.stdout, sys.stderr = out, err
        os.chdir(spec["runtime_root"])
        source = str(Path(spec["source_root"]) / "src")
        os.environ["PYTHONPATH"] = source
        sys.path.insert(0, source)
        sys.argv = ["btc_futures_bot.main", *spec["arguments"]]
        try:
            runpy.run_module("btc_futures_bot.main", run_name="__main__")
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == "__main__":
    main()
