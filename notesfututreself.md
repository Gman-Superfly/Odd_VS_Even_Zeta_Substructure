run dis ting note ofrm y future self

run on downstairs windows machine

powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_all_tests.ps1

- Use Python 3.10+; manage deps with uv:
  - Install uv (Windows PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
  - Sync deps: `uv sync`
  - Run tests: `uv run -m pytest -q`
- Match style: PEP8 + type hints; keep functions small and single-purpose.
- Add unit tests for new functions (see `tests/`).
- For performance features (GPU, profiling), provide before/after numbers.


