# Live Parser Structure: Ordered Form Elements

## Problem Statement

Currently, forms in the colight-site parser/generator system don't maintain the order of their elements (prose, statements, expressions). The original implementation stored prose and code separately, losing the relative ordering within a form.

Key issues:
1. Forms don't maintain an ordered list of their elements
2. Element order is lost between prose and code
3. The data flow from parser → generator → JSON → React doesn't preserve order

## Current Structure

### Parser (parser.py)
- `Form` class has:
  - `markdown`: List of comment lines (prose)
  - `node`: A CST node or CombinedCode (mixing statements/expressions)
  - `is_expression` property: Checks if the entire form is an expression
  - `is_statement` property: Inverse of is_expression (excluding dummy forms)
  - Pragma handling mixed throughout

### Generator (generator.py)
- Processes forms sequentially
- Applies pragma rules differently for prose vs code
- Visual display tied to `form.is_expression`

### JSON Generator (json_generator.py)
- Creates a `content` array but doesn't clearly separate element types
- Still relies on form-level `is_expression` for visual display

### React (live.jsx)
- Renders content items but doesn't have access to the ordered structure
- Can't make proper decisions about which element determines visual display

## Proposed Changes

### 1. New Data Model

Create a clear element-based structure:

```python
@dataclass
class FormElement:
    type: Literal["prose", "statement", "expression"]
    content: Union[str, cst.CSTNode]  # str for prose, CSTNode for code
    line_number: int

@dataclass
class Form:
    elements: List[FormElement]  # Ordered list of all elements
    metadata: FormMetadata       # Contains pragma set for the form
    start_line: int
    
    def should_show_element(self, element: FormElement) -> bool:
        """Determine if element should be shown based on form pragma."""
        if element.type == "prose":
            return not should_hide_prose(self.metadata.pragma)
        elif element.type in ("statement", "expression"):
            # Both statements and expressions are code
            return not should_hide_code(self.metadata.pragma)
        return True
    
    def should_show_visual(self) -> bool:
        """Check if visuals should be shown for expressions in this form."""
        return not should_hide_visuals(self.metadata.pragma)
```

### 2. Parser Changes

Modify the parser to build ordered elements:

1. During form construction, maintain element order
2. Track whether each code node is a statement or expression
3. Keep prose as separate elements

### 3. Generator Changes

Simplify the generator to iterate over elements:

1. For each element, check if it should be shown using `form.should_show_element(elem)`
2. For expressions, also check `form.should_show_visual()` to determine if visual output should be displayed
3. Handle special case where `hide-statements` hides statements but not expressions

### 4. JSON Structure

Output cleaner JSON with explicit element ordering:

```json
{
  "forms": [{
    "id": 0,
    "pragma": ["hide-statements", "show-visuals"],
    "elements": [
      {"type": "prose", "content": "This is a comment", "show": true},
      {"type": "statement", "content": "x = 1", "show": false},
      {"type": "expression", "content": "x + 1", "show": true, "showVisual": true}
    ]
  }]
}
```

### 5. React Changes

Update React to understand the element structure and make display decisions based on element types and positions.

## Key Design Principles

1. **Forms are evaluation units** - All code in a form is evaluated together
2. **Pragmas are per-form** - Multiple pragma lines accumulate into a set
3. **Visibility is per-element** - Based on element type and form pragma
4. **Any expression can show a visual** - Not restricted to last element
5. **Code = statements + expressions** - Both are code, expressions can produce visuals

## Visibility Rules

For each element in a form:
- **Prose**: Show unless `hide-prose` is in pragma
- **Statement**: Show unless `hide-code` OR `hide-statements` is in pragma  
- **Expression**: Show unless `hide-code` is in pragma
- **Visual** (for expressions): Show unless `hide-visuals` is in pragma

Note: `show-*` pragmas always override corresponding `hide-*` pragmas.

## Implementation Plan

1. **Update Form class** with positive visibility methods
2. **Fix element creation** in parser to properly classify statements vs expressions
3. **Update tests** to match actual behavior (blank lines break forms)
4. **Update generator** to use ordered elements and visibility rules
5. **Update JSON generator** to include visibility flags
6. **Update React** to respect visibility flags
7. **Run all tests** to ensure backward compatibility

## Benefits

1. **Clarity**: Forms as evaluation units, elements as content
2. **Correctness**: Proper element ordering preserved
3. **Flexibility**: Any expression can show visuals
4. **Future-proof**: Ready for dependency tracking between forms
5. **Positive logic**: `should_show_*` methods are clearer