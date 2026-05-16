# CLAUDE.md

## Coding Guidelines

1. **Always use type hints** on function signatures (parameters and return types).

2. **Use self-explanatory variable names.** Avoid truncated names and abstract acronyms. Domain-standard abbreviations are acceptable when widely understood by the target audience (e.g., `cv`, `df`, `acf`, `pca`, `hpo`).

3. **Prefer the simplest solution that works** while adhering to SOLID principles. Only introduce abstractions when there is a concrete need, not a hypothetical future one.

4. **Always write docstrings** for public functions and classes. Use Google-style docstrings with `Args` and `Returns` sections.

5. **Write unit tests for all public functions.** Tests live in the `tests/` directory and use pytest.

6. **Keep notebooks free of function definitions.** Functions belong under `src/` or `scripts/`.

## Project Structure

- Always include a `.gitignore`.
- Always include a `README.md` that recommends creating a conda environment named after the repo.
- Always maintain a `requirements.txt` reflecting the imports used in the repo.

## Dependencies

- Optional dependencies are guarded with try/except at import time (e.g., `ruptures`, `lightgbm`, `optuna`, `scikit-optimize`).
- Public functions that require optional deps call a guard function (e.g., `_check_ruptures()`) as their first line, raising a helpful `ImportError` with install instructions.

## Running Tests

```bash
pytest tests/
```
