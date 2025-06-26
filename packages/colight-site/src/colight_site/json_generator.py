"""Generate JSON representation of forms for LiveServer."""

import json
import base64
import pathlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .parser import Form, parse_colight_file
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

        # Merge pragma tags
        combined_pragma_tags = self.config.pragma_tags.union(file_metadata.pragma_tags)

        # Execute forms
        executor = SafeFormExecutor(verbose=self.config.verbose)
        form_results = []

        for i, form in enumerate(forms):
            try:
                result = executor.execute_form(form, str(source_path))
                colight_bytes = executor.get_colight_bytes(result)
                form_results.append(
                    {"value": result, "visual_data": colight_bytes, "error": None}
                )
            except Exception as e:
                form_results.append(
                    {
                        "value": None,
                        "visual_data": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        # Build JSON structure
        doc = {
            "file": str(source_path.name),
            "metadata": {
                "pragma_tags": sorted(list(combined_pragma_tags)),
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
        # Resolve pragma tags
        form_tags = form.metadata.resolve_with_defaults(self.config.pragma_tags)

        # Build content array
        content = []

        # Add markdown content if present
        markdown_text = "\n".join(form.markdown).strip()
        if markdown_text:
            content.append({"type": "markdown", "value": markdown_text})

        # Skip code generation for dummy forms
        if not form.is_dummy_form:
            # Get the code for this form
            code = form.code.strip()

            if code:
                # Keep all code together in a single block
                content.append(
                    {
                        "type": "code",
                        "value": code,
                        "isStatement": form.is_statement,
                        "isExpression": form.is_expression,
                    }
                )

        # Add error if present
        if result["error"]:
            content.append({"type": "error", "value": str(result["error"])})

        # Add visual if present and it's an expression
        if form.is_expression and result["visual_data"] is not None:
            visual_item = self._serialize_visual(result["visual_data"])
            if visual_item:
                content.append(visual_item)

        # Skip empty forms
        if not content:
            return None

        # Determine end type
        if form.is_dummy_form:
            end_type = "markdown"
        elif form.is_expression:
            end_type = "expression"
        else:
            end_type = "statement"

        return {
            "id": form_id,
            "line": form.start_line,
            "pragmaTags": sorted(list(form_tags)),
            "content": content,
            "endType": end_type,
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
