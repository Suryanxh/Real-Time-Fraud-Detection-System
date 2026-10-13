import pandas as pd

from src.features import add_merchant_risk, synthesize_entity_fields


def test_synthetic_entities_do_not_depend_on_fraud_labels():
    base = pd.DataFrame({"Time": [1, 2, 3, 4], "Class": [0, 1, 0, 1]})
    flipped = base.assign(Class=1 - base["Class"])

    first = synthesize_entity_fields(base, seed=42)
    second = synthesize_entity_fields(flipped, seed=42)

    pd.testing.assert_frame_equal(
        first[["merchant", "device_id", "ip"]],
        second[["merchant", "device_id", "ip"]],
    )


def test_merchant_risk_feature_is_constant_base_rate():
    transactions = pd.DataFrame(
        {"merchant": ["shop_a", "shop_a", "shop_b", "shop_b"], "Class": [0, 0, 1, 1]}
    )

    features, risk_map, base_rate = add_merchant_risk(transactions, is_train=True)

    assert base_rate == 0.5
    assert risk_map == {"shop_a": 0.5, "shop_b": 0.5}
    assert features["merchant_risk_score"].tolist() == [0.5] * 4
