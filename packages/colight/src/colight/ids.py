"""Per-run ("volatile") identifiers and the pattern that recognizes them.

Widget ids and generated state keys are regenerated on every run, so they
must never leak into fingerprints or diffs. Every mint site lives here, next
to :data:`VOLATILE_ID_RE`, which the CLI's canonicalization uses to replace
them with stable placeholders — if the id scheme changes, the coupling test
in ``tests/test_ids.py`` fails loudly.
"""

import re
import uuid

WIDGET_ID_PREFIX = "colight-widget-"

# Matches every id minted below: widget ids and bare uuid state keys.
VOLATILE_ID_RE = re.compile(
    r"(?:colight-widget-[0-9a-f]{32}"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def widget_id() -> str:
    """Mint a widget instance id (regenerated per run)."""
    return f"{WIDGET_ID_PREFIX}{uuid.uuid4().hex}"


def state_key() -> str:
    """Mint a generated state key for collected state entries."""
    return str(uuid.uuid4())


def ref_state_key() -> str:
    """Mint a generated state key for ``Ref`` values."""
    return str(uuid.uuid1())


__all__ = [
    "VOLATILE_ID_RE",
    "WIDGET_ID_PREFIX",
    "ref_state_key",
    "state_key",
    "widget_id",
]
