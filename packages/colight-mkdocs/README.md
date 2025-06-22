# colight-mkdocs

MkDocs plugins for Colight documentation.

## Features

- **API Documentation Plugin**: Automatically generates API documentation from Python modules
- **Site Plugin**: Processes `.py` and `.colight.py` files for MkDocs sites

## Installation

```bash
pip install colight-mkdocs
```

## Usage

Add to your `mkdocs.yml`:

```yaml
plugins:
  - colight.mkdocs.api_plugin:
      modules:
        - module: colight.plot
          output: api/plot.md
  - colight.mkdocs.site_plugin:
      format: markdown
      include:
        - "*.py"
        - "*.colight.py"
```
