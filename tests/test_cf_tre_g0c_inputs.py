from experiments.sanitize_cf_tre_g0c_inputs import _assert_no_test_key


def test_sanitizer_rejects_nested_test_keys():
    try:
        _assert_no_test_key({"safe": {"test_metrics": 1}})
    except ValueError as exc:
        assert "forbidden test key" in str(exc)
    else:
        raise AssertionError("nested test key was accepted")


def test_sanitizer_accepts_selection_and_validation_names():
    _assert_no_test_key({"trial_split_validation": [1], "components": {"linear": {"C": 1.0}}})
