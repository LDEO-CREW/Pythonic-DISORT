<!-- omit in toc -->
# Contributing to PythonicDISORT

Thanks for contributing — improvements to correctness, performance, docs, and examples are all welcome.

**Quick links**
- Docs: https://pythonic-disort.readthedocs.io/en/latest/
- Issues: https://github.com/LDEO-CREW/Pythonic-DISORT/issues
- Security concerns: email dh3065@columbia.edu (please do not open a public issue)

<!-- omit in toc -->
## Table of contents
- [Getting set up](#getting-set-up)
- [Running tests](#running-tests)
- [Reporting bugs](#reporting-bugs)
- [Requesting changes](#requesting-changes)
- [Contributing code](#contributing-code)
- [Improving documentation and notebooks](#improving-documentation-and-notebooks)
- [License](#license)

## Getting set up

PythonicDISORT targets **Python 3.8+**.

1) Fork the repo and clone your fork:

```bash
git clone https://github.com/<your-username>/Pythonic-DISORT.git
cd Pythonic-DISORT
```

2) Create and activate a virtual environment:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

3) Install the project in editable mode.

- Minimal runtime deps (installs `numpy` + `scipy`)  
  ```bash
  pip install -e .
  ```

- Dev/test deps (adds `pytest`)  
  ```bash
  pip install -e ".[pytest]"
  ```

- Notebook/example deps (adds `autograd`, `jupyter`, `notebook`, `matplotlib`)  
  ```bash
  pip install -e ".[notebook_dependencies]"
  ```

## Running tests

The repo contains verification tests under `pydisotest/`.

Run them like this:

```bash
pip install -e ".[pytest]"
cd pydisotest
pytest
```

Some of the deeper verification work (notably the later parts of the big documentation notebook) compares against a **wrapped Fortran DISORT** or equivalent. If you don’t have that locally, you can still run most tests and contribute meaningfully — just call it out in your PR.

## Reporting bugs

Before opening a new issue:
- Search existing issues (including closed ones).
- Make sure you’re on a recent release.
- Reduce the problem to a minimal reproducible example.

When you open an issue, include:
- What you expected vs what happened.
- A minimal script or notebook cell that reproduces it.
- Your environment: OS, Python version, numpy/scipy versions.
- If relevant: number of layers, number of streams, phase function choice, boundary conditions, and any unusual parameter ranges.

### Security

If the bug involves security, privacy, or accidental disclosure (e.g., credentials in logs), **email dh3065@columbia.edu** instead of filing an issue.

## Requesting changes

Feature requests are welcome, but this project is primarily a **numerical/scientific** codebase — proposals should be specific.

A good request includes:
- The use case and why existing functionality isn’t enough.
- A concrete API sketch (function signature, inputs/outputs).
- Any relevant references (papers, equations, DISORT behavior you’re matching).
- A suggestion for how to test/verify it.

For larger changes, please open an issue first so effort doesn’t get wasted.

## Contributing code

### Ground rules
- Keep PRs small and focused.
- Don’t add heavyweight new dependencies lightly. If a dependency is optional, keep it optional.
- If you change numerical behavior, explain *why* and include a verification test.

### Workflow
1) Create a branch from `main`:
   ```bash
   git checkout -b your-branch-name
   ```
2) Make your change with tests.
3) Run:
   - `pytest` from `pydisotest/` (see above)
4) Open a PR:
   - Describe what changed and why.
   - Link the related issue (if any).
   - Mention what you tested locally (and what you couldn’t).

### Style
There’s no strict formatter enforced in this repo. Aim for:
- PEP 8-ish readability
- Clear variable names (this is math-heavy code; clarity beats cleverness)
- Docstrings for user-facing functions

## Improving documentation and notebooks

The docs are hosted on Read the Docs, and the repository includes a comprehensive Jupyter notebook (`docs/Pythonic-DISORT.ipynb`) that serves as extended documentation, derivations, and verification.

If you edit notebooks or examples, please:
- Keep outputs deterministic (avoid random seeds unless fixed).
- Prefer smaller, faster-running cells when possible.
- If you add a new example, explain the physical meaning of inputs/outputs.

For local builds of the Sphinx docs (if you’re editing them), install whatever the docs build requires (often `sphinx`/`nbsphinx` plus the notebook extras), then build from the repo root. If you’re unsure, open a PR and we’ll help you get it building.

## License

By contributing, you agree that your contributions will be licensed under the project’s MIT license.
