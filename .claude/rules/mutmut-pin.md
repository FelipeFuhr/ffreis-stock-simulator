---
paths:
  - "pyproject.toml"
  - ".github/workflows/mutation.yml"
  - "Makefile"
---

## mutmut version pin

- **`mutmut` is pinned `>=2.4,<3`** in the `dev` extra — mutmut 3.x dropped the
  `--paths-to-mutate`/`--tests-dir` CLI flags that `make mutation-test` and
  `.github/workflows/mutation.yml` (via `python-mutation.yml`) still invoke.
  Un-pin only after that reusable workflow migrates to the `[tool.mutmut]`
  config-table API.
