"""Generate JSON representation of forms for LiveServer."""

import json
import base64
import pathlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .parser import Form, parse_colight_file, should_show_prose, should_show_code, should_show_statements, should_show_visuals
from .executor import SafeFormExecutor
from .builder import BuildConfig

@dataclass
class JsonFormGenerator:
    """Convert parsed forms to JSON representation for client-side rendering."""

    config: BuildConfig = field(default_factory=BuildConfig)

    def generate_json(self, source_path: pathlib.Path) -> str:
        """Generate JSON representation of a Python file."""
        # Parse the file
        forms, file_metadata = parse_colight_file(source_path)

        pragma = self.config.pragma | file_metadata.pragma

        # Execute forms
        executor = SafeFormExecutor(verbose=self.config.verbose)
        form_results = []

        for i, form in enumerate(forms):
            try:
                result = executor.execute_form(form, str(source_path))
                colight_bytes = executor.get_colight_bytes(result)
                form_results.append(
                    {"value": result, "result": colight_bytes, "error": None}
                )
            except Exception as e:
                form_results.append(
                    {
                        "value": None,
                        "result": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        # Build JSON structure
        doc = {
            "file": str(source_path.name),
            "metadata": {
                "pragma": sorted(list(pragma)),
                "title": source_path.stem,
            },
            "forms": [],
        }

        # Convert each form to JSON
        for i, (form, result) in enumerate(zip(forms, form_results)):
            json_form = self._form_to_json(form, result, i)
            if json_form:  # Skip empty forms
                doc["forms"].append(json_form)

        return json.dumps(doc, indent=2)

    def _form_to_json(
        self, form: Form, result: Dict[str, Any], form_id: int
    ) -> Optional[Dict[str, Any]]:
        """Convert a single form to JSON representation."""
        
        form_tags = form.metadata.resolve_with_defaults(self.config.pragma)

        # Build elements array with visibility flags
        elements = []
        
        # Process each element in order
        for elem in form.elements:
            elem_data = {"type": elem.type}
            
            if elem.type == "prose":
                prose_text = elem.content if isinstance(elem.content, str) else ""
                elem_data["value"] = prose_text.strip()
                elem_data["show"] = should_show_prose(form_tags)
                
            elif elem.type in ("statement", "expression"):
                code = elem.get_code().strip()
                elem_data["value"] = code
                
                # Determine visibility
                if elem.type == "statement":
                    elem_data["show"] = should_show_code(form_tags) and should_show_statements(form_tags)
                else:  # expression
                    elem_data["show"] = should_show_code(form_tags)
            
            # Skip empty elements
            if elem_data.get("value"):
                elements.append(elem_data)
        
        # Add visual data to last element if it's an expression
        if (form.last_element and form.last_element.type == "expression" 
            and result["result"] is not None and should_show_visuals(form_tags)):
            # Find the last expression element in our elements array
            for i in range(len(elements) - 1, -1, -1):
                if elements[i]["type"] == "expression":
                    visual_item = self._serialize_visual(result["result"])
                    if visual_item:
                        elements[i]["visual"] = visual_item
                        elements[i]["showVisual"] = True
                    break

        # Add error if present
        if result["error"]:
            elements.append({
                "type": "error", 
                "value": str(result["error"]),
                "show": True
            })

        # Skip empty forms
        if not elements:
            return None

        return {
            "id": form_id,
            "line": form.start_line,
            "pragma": sorted(list(form_tags)),
            "elements": elements,
            "hasExpression": form.is_expression,
            "showsVisual": form.is_expression and should_show_visuals(form_tags)
        }

    def _serialize_visual(self, visual_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Serialize a visual output to JSON."""
        if not visual_bytes:
            return None

        try:
            # Encode the bytes as base64
            encoded = base64.b64encode(visual_bytes).decode("utf-8")

            return {
                "type": "visual",
                "format": "inline",
                "size": len(visual_bytes),
                "data": encoded,
            }
        except Exception:
            # If encoding fails, don't include it
            return None
