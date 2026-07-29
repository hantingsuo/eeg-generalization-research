from experiments.audit_seediv_implementation_consistency import (
    function_literal_assignments,
)


def test_function_literal_assignments_is_scoped_to_named_function():
    source = """
def first():
    labels = [1, 2]

def second():
    labels = [3, 4]
    files = [['a.mat']]
"""
    assert function_literal_assignments(
        source,
        "second",
        ("labels", "files"),
    ) == {"labels": [3, 4], "files": [["a.mat"]]}
