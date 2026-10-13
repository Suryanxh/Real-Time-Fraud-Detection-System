import pandas as pd

from src import stream


def test_demo_sample_contains_both_classes_in_timestamp_order(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "Class": [0, 1, 0, 1, 0, 1],
            "Amount": [20, 10, 100, 30, 5, 40],
            "timestamp": [
                "2024-01-01 00:00:05",
                "2024-01-01 00:00:04",
                "2024-01-01 00:00:03",
                "2024-01-01 00:00:02",
                "2024-01-01 00:00:01",
                "2024-01-01 00:00:00",
            ],
        }
    )
    path = tmp_path / "test_features.csv"
    data.to_csv(path, index=False)
    monkeypatch.setattr(stream, "TEST_DATA_PATH", str(path))

    sample = stream.load_test_data(limit=4)

    assert sample["Class"].value_counts().to_dict() == {0: 2, 1: 2}
    assert sample["timestamp"].tolist() == sorted(sample["timestamp"].tolist())
    assert 100 in sample.loc[sample["Class"] == 0, "Amount"].tolist()
