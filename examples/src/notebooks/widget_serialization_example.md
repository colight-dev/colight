# Widget Serialization Extension Example

The widget serialization system is now extensible. Here's how to add custom serializers:

```python
from colight.widget import register_serializer, SKIP

# Register a custom serializer for a specific type
@register_serializer("my_custom_type")
def serialize_my_type(data, collected_state):
    """Serializer for MyCustomType objects."""
    if not isinstance(data, MyCustomType):
        return SKIP  # Let other serializers handle it

    return {
        "__type__": "my_custom",
        "value": data.to_dict()
    }

# Control serializer order
@register_serializer("high_priority", before="numpy_array")
def serialize_special_array(data, collected_state):
    """Run before numpy serializer to handle special cases."""
    if hasattr(data, 'special_flag'):
        return {"__type__": "special", "data": data.tolist()}
    return SKIP

# List all registered serializers
from colight.widget import list_serializers
print(list_serializers())
# Output: ['collector', 'high_priority', 'numpy_array', 'for_json', 'attributes_dict', 'callable', 'my_custom_type']
```

## Design Notes

- Fast paths (str, int, bool, None, dict, list) remain in if/elif for performance
- Custom serializers run after fast paths but before the TypeError
- Serializers must return SKIP (object identity) to pass to the next serializer
- Order matters - use before/after parameters to control precedence
- Registration happens at import time with zero runtime cost
