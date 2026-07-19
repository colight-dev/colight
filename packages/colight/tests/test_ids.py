"""Coupling test for volatile id minting vs canonicalization.

Fingerprints and diffs rely on ``colight.ids.VOLATILE_ID_RE`` recognizing
every per-run id the library mints. If any mint site drifts from the
pattern, these tests fail loudly instead of fingerprints silently becoming
run-dependent.
"""

import colight.ids as ids
import colight.plot as Plot
from colight.layout import Ref
from colight.plot_spec import MarkSpec


def test_minting_helpers_match_pattern():
    for minted in (ids.widget_id(), ids.state_key(), ids.ref_state_key()):
        assert ids.VOLATILE_ID_RE.fullmatch(minted), minted


def test_real_mint_sites_match_pattern():
    # Widget id (layout.LayoutItem.get_id).
    widget_id = Plot.dot({"x": [1.0], "y": [2.0]}).get_id()
    assert ids.VOLATILE_ID_RE.fullmatch(widget_id), widget_id
    # MarkSpec state key (plot_spec.MarkSpec).
    mark_key = MarkSpec("dot", {"x": [1.0]}, {})._state_key
    assert ids.VOLATILE_ID_RE.fullmatch(mark_key), mark_key
    # Generated Ref state key (layout.Ref).
    ref_key = Ref([1, 2])._state_key
    assert ids.VOLATILE_ID_RE.fullmatch(ref_key), ref_key


def test_explicit_state_keys_are_not_volatile():
    # User-chosen keys (Plot.State) must not be canonicalized away.
    assert not ids.VOLATILE_ID_RE.fullmatch("my_key")
    ref = Ref(1, state_key="my_key")
    assert ref._state_key == "my_key"
