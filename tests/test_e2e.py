from __future__ import annotations

import json

from dysemkt.engine import TrainConfig, train
from dysemkt.preprocess import preprocess_moocradar
from dysemkt.text import HashTextEncoder


def test_end_to_end_training_writes_checkpoint_and_metrics(raw_moocradar, tmp_path):
    data_dir = tmp_path / "processed"
    output_dir = tmp_path / "run"
    preprocess_moocradar(raw_moocradar, data_dir, HashTextEncoder(16))
    config = TrainConfig(
        d_model=16, history_length=4, batch_size=16, epochs=1, patience=1,
        feature_mode="semantic",
    )
    result = train(data_dir, output_dir, config)
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert 0.0 <= result["test"]["accuracy"] <= 1.0
    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["split_counts"]["train"] > 0

