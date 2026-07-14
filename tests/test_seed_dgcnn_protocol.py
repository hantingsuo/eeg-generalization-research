import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).parents[1] / "experiments" / "run_seed_dgcnn_subject_split.py"
    spec = importlib.util.spec_from_file_location("run_seed_dgcnn_subject_split", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_libeer_subject_split_matches_seed_2024_reference():
    runner = _load_runner()
    split = runner.libeer_subject_split(2024)
    assert split["test"] == [8, 0, 5]
    assert split["validation"] == [13, 12, 1]
    assert split["train"] == [14, 10, 6, 3, 4, 9, 11, 2, 7]
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert set().union(*map(set, split.values())) == set(range(15))
