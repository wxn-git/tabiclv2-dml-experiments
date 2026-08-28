from tabdml.validation import compare_with_doubleml


def test_custom_estimator_matches_doubleml_with_identical_predictions():
    result = compare_with_doubleml(n=300, seed=123)
    assert result["theta_difference"] < 1e-10
    assert result["standard_error_difference"] < 1e-10

