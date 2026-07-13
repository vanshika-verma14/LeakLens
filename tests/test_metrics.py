"""Tests for the recovery metric — the most important test in the project, because it
pins down what "leaked" means. Fast: strings + hand-made vectors, no model load.

Every assertion is something we'd defend out loud (DECISIONS D5). The common-word-label
tests document a real limitation of case-insensitive substring matching on purpose.
"""
from leaklens.inversion import metrics


# --- entity_found: mode semantics -------------------------------------------------

def test_ci_catches_recapitalization_that_exact_misses():
    # Real T1.2 case: 'library' recovered as 'Library'.
    recovered = "the Library holds its exam season"
    assert metrics.entity_found("library", recovered, mode="exact") is False
    assert metrics.entity_found("library", recovered, mode="ci") is True


def test_fuzzy_does_not_count_a_wrong_high_entropy_value():
    # Real pii case: real phone 555-0112 mangled to a DIFFERENT number 555-0291.
    recovered = "call 555-0291 to confirm"
    assert metrics.entity_found("555-0112", recovered, mode="exact") is False
    assert metrics.entity_found("555-0112", recovered, mode="ci") is False
    # fuzzy at the default 0.8 must NOT count the wrong number as recovered (ratio ~0.75).
    assert metrics.entity_found("555-0112", recovered, mode="fuzzy") is False


def test_empty_entity_never_found():
    assert metrics.entity_found("", "anything at all", mode="ci") is False


# --- key_entity_recall: full / partial / zero -------------------------------------

def test_full_recall_low_entropy():
    recovered = "recommendation engine soared due to the redesign"
    recall, matched, missed = metrics.key_entity_recall(
        ["recommendation engine", "redesign"], recovered, mode="ci")
    assert recall == 1.0 and missed == []


def test_partial_recall_pii():
    recovered = "Call Aisha Khan on 555-0291 to confirm"
    recall, matched, missed = metrics.key_entity_recall(
        ["Aisha Khan", "555-0112"], recovered, mode="ci")
    assert recall == 0.5
    assert matched == ["Aisha Khan"] and missed == ["555-0112"]


def test_zero_recall_high_entropy_credential():
    recovered = "evoke key repleasure (Swave RSh-9999-E-0) for the deployment"
    recall, _, _ = metrics.key_entity_recall(
        ["deploy key", "ssh-ed25519-EXAMPLE-9ee11b"], recovered, mode="ci")
    assert recall == 0.0


def test_empty_key_entities_is_zero_recall():
    recall, matched, missed = metrics.key_entity_recall([], "whatever", mode="ci")
    assert recall == 0.0 and matched == [] and missed == []


# --- common-word label false positive: DOCUMENTED behavior, not a surprise --------

def test_common_word_label_matches_coincidentally():
    """A single common-word label ('Transaction') ci-matches generic text that did NOT
    recover the specific record. This overstates leakage — documented limitation."""
    recovered = "the transaction was declined earlier today"
    # the label coincidentally matches...
    assert metrics.entity_found("Transaction", recovered, mode="ci") is True
    # ...but the high-entropy value (the honest signal) is correctly absent.
    assert metrics.entity_found("TXN-88213", recovered, mode="ci") is False


def test_common_word_label_inflates_row_recall_but_value_is_the_truth():
    """score_row shows the mechanism: the label lifts recall to 0.5 even though the
    actual transaction (its value) was never recovered. This is why labels and values
    are tagged separately — read value recovery, not just the headline number."""
    row = {"id": "struct-x", "type": "structured",
           "key_entities": ["Transaction", "TXN-88213"]}
    score = metrics.score_row(row, "the transaction was declined earlier today", mode="ci")
    assert score.recall == 0.5
    assert score.matched == ["Transaction"]      # coincidental label hit
    assert score.missed == ["TXN-88213"]         # the value — the real test — is missed


# --- aggregation + cosine ---------------------------------------------------------

def test_score_row_records_all_three_mode_details():
    row = {"id": "p", "type": "plain", "key_entities": ["library"]}
    score = metrics.score_row(row, "the Library holds", mode="ci")
    assert score.recall == 1.0                    # ci (primary)
    assert score.details["exact"] == 0.0          # exact would miss the recapitalization
    assert score.details["ci"] == 1.0
    assert score.cosine is None                   # no embeddings supplied


def test_per_category_recall_groups_by_type():
    scores = [
        metrics.RecoveryScore("a", "plain", 1.0, 1, 1, [], []),
        metrics.RecoveryScore("b", "plain", 0.0, 0, 1, [], []),
        metrics.RecoveryScore("c", "credential", 0.0, 0, 2, [], []),
    ]
    assert metrics.per_category_recall(scores) == {"credential": 0.0, "plain": 0.5}


def test_cosine_identical_and_orthogonal():
    assert metrics.cosine_similarity([1, 2, 3], [1, 2, 3]) == 1.0
    assert metrics.cosine_similarity([1, 0], [0, 1]) == 0.0
    assert metrics.cosine_similarity([0, 0], [1, 1]) == 0.0   # zero vector -> 0, no crash


def test_recall_distribution_is_threshold_free():
    scores = [
        metrics.RecoveryScore("a", "plain", 1.0, 2, 2, [], []),
        metrics.RecoveryScore("b", "pii", 0.5, 1, 2, [], []),
        metrics.RecoveryScore("c", "credential", 0.0, 0, 2, [], []),
    ]
    dist = metrics.recall_distribution(scores)
    assert set(dist) == {"overall", "by_category"}
    assert set(dist["by_category"]) == {"plain", "pii", "credential"}
    assert dist["overall"]["max"] == 1.0 and dist["overall"]["min"] == 0.0
    # no key anywhere names a threshold — the human sets that later
    assert "threshold" not in str(dist).lower()
