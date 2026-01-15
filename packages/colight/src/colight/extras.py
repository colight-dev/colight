from colight.layout import JSRef

bylight = Bylight = JSRef("Bylight")
"""Creates a highlighted code block using the [Bylight library](https://mhuebert.github.io/bylight/).

Args:
    source (str): The source text/code to highlight
    patterns (list): A list of patterns to highlight. Each pattern can be either:
        - A string to match literally
        - A dict with 'match' (required) and 'color' (optional) keys
    props (dict, optional): Additional properties to pass to the pre element. Defaults to {}.

Example:
    ```python
    import colight.plot as Plot
    from colight import extras

    extras.bylight('''
        def hello():
            print("Hello World!")
    ''', ["def", "print"])
    ```

Returns:
    A Bylight component that renders the highlighted code block.
"""

__all__ = ["bylight", "Bylight"]
