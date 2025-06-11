# Developer's Guide

This guide covers common development tasks in the Colight codebase.

### Jupyter notes

A typical and recommended workflow is to use colight with VS Code's Python Interactive Window. With the VS Code jupyter extension installed, one can use ordinary `.py` files with `# %%` markers to separate cells, then run the `Jupyter: Run Current Cell` command. Results, including plots, will be rendered with VS Code.

Of course, one can also use colight from within Jupyter Labs and Colab.

If jupyter has trouble finding a kernel to evaluate from, you can install one (using poetry) via:

```bash
poetry run python -m ipykernel install --user --name colight
```

### Pre-commit Hooks

Pre-commit hooks ensure code consistency. They run automatically on each commit to format Python code and perform other checks.

Setup:

1. Install pre-commit:
```bash
pipx install pre-commit
```

2. Install hooks:
```bash
pre-commit install
```

Run hooks manually:
```bash
pre-commit run --all-files
```

Hooks are configured in `.pre-commit-config.yaml`.


### WebGPU Testing

The scene3d features require a webgpu-supported environment for testing, so there is an extra GitHub action to demonstrate/test webgpu screenshot capture in CI. Note that we specify a gpu-enabled runner for this action.
