"""Golden round-trip test for the vec2text inverter.

This asserts *honest* behavior, not an idealized one. Phase-0 showed high-entropy
secrets mangle while low-entropy text and named entities survive (DECISIONS D5), so we
must NOT assert exact-match recovery. Instead: take one low-entropy plain sentence with
known key entities, invert it, and assert that MOST of those entities resurface in the
recovered text.

Real inversion is slow on CPU, so this is marked `slow` — run it with `pytest --runslow`.
"""
import pytest

from leaklens.inversion import inverter

# A low-entropy plain sentence (mirrors a corpus 'plain' row) with known content words.
SENTENCE = "The library extended its opening hours during the exam season."
KEY_ENTITIES = ["library", "opening hours", "exam season"]
RECOVERY_FLOOR = 0.6          # "most" of the key entities — calibrated by running
TEST_NUM_STEPS = 5            # fewer steps than the default 20, for CPU speed


def test_inverter_constructs_without_loading():
    """Building an Inverter is cheap and exposes the locked model ids (no download)."""
    inv = inverter.Inverter()
    assert inv.encoder_name == inverter.ENCODER
    assert inv.inversion_name == inverter.INVERSION
    assert inv.corrector_name == inverter.CORRECTOR
    assert inv._corrector is None  # nothing loaded yet


@pytest.mark.slow
def test_low_entropy_recovers_most_key_entities():
    recovered = inverter.get_inverter().roundtrip([SENTENCE], num_steps=TEST_NUM_STEPS)[0].lower()
    hits = sum(ent.lower() in recovered for ent in KEY_ENTITIES)
    frac = hits / len(KEY_ENTITIES)
    assert frac >= RECOVERY_FLOOR, (
        f"only {hits}/{len(KEY_ENTITIES)} key entities recovered "
        f"(floor {RECOVERY_FLOOR}); recovered={recovered!r}"
    )
