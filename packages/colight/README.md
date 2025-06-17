# Colight

A Python library for creating interactive visualizations and widgets.

## Installation

```bash
pip install colight
```

## Usage

```python
import colight as cl
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create visualization
cl.plot.line(x=x, y=y)
```

For more examples and documentation, visit the [documentation site](https://colight.dev).
