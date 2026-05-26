# CLAUDE.md

## Coding Guidelines

1. **Always use type hints** on function signatures (parameters and return types).

2. **Use self-explanatory variable names.** Avoid truncated names and abstract acronyms. Domain-standard abbreviations are acceptable when widely understood by the target audience (e.g., `cv`, `df`, `acf`, `pca`, `hpo`).

3. **Prefer the simplest solution that works** while adhering to SOLID principles. Only introduce abstractions when there is a concrete need, not a hypothetical future one.

4. **Always write docstrings** for public functions and classes. Use Google-style docstrings with `Args` and `Returns` sections.

5. **Write unit tests for all public functions.** Tests live in the `tests/` directory and use pytest.

6. **Keep notebooks free of function definitions.** Functions belong under `src/` or `scripts/`.

