# colight.vscode: Rapid Python Eval

Evaluate Python in an [interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py#_using-the-python-interactive-window) with automatic detection of "cell" boundaries - zero markup. Enjoy notebook-like feedback in plain python files.

By default `ctrl-enter` evaluates the current cell and stays there, `shift-enter` evaluates & moves to the next cell.

A "cell" is defined as:
* the current line, function, or class, plus:
* any adjacent cells not separated by a blank line

![Colight Demo](images/demo.gif)

## Features

- Automatically detect code cells based on function/class, contiguous regions, and configurable cell markers. No need to pollute your python files with cell metadata.
- Supports both code and markdown cells
- Visual feedback with cell highlighting
- Compatible with Jupyter notebooks and Python scripts using [Jupytext "light" format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-light-format)
- Integrates with VS Code's Python Interactive Window

## Requirements

- Visual Studio Code v1.90.0 or higher
- Python extension for VS Code

## Extension Settings

This extension contributes the following settings:

* `colight.enabledCellMarkers`: An array of strings that define the cell markers. Default: `["# +", "# %%"]`
* `colight.useExplicitCellsIfPresent`: If true, use explicit cell markers when present.
* `colight.renderComments`: If true, renders comments as Markdown.
* `colight.currentCell.borderWidth`: Border width for the current cell decoration.
* `colight.currentCell.show`: Decorate the current cell with a border above/below.
* `colight.debugMode`: Enable debug logging.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This extension is licensed under the [MIT License](LICENSE).
