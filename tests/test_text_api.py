from __future__ import annotations

import numpy as np

from dysemkt.cli import build_parser
from dysemkt.text import APITextEncoder, load_env_file


class FakeAPITextEncoder(APITextEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.payloads = []

    def _post_json(self, path, payload):
        self.calls += 1
        self.payloads.append((path, payload))
        return {
            "data": [
                {"index": index, "embedding": [float(index), float(len(text))]}
                for index, text in reversed(list(enumerate(payload["input"])))
            ]
        }


def test_api_text_encoder_orders_rows_and_sets_dimension(tmp_path):
    encoder = FakeAPITextEncoder(
        "text-embedding-test", batch_size=2, base_url="https://example.test/v1", cache_dir=tmp_path,
    )
    values = encoder.encode(["alpha", "b"])

    np.testing.assert_array_equal(values, np.asarray([[0.0, 5.0], [1.0, 1.0]], dtype=np.float32))
    assert encoder.dimension == 2
    assert encoder.calls == 1
    assert encoder.payloads[0][0] == "/embeddings"
    assert encoder.payloads[0][1]["model"] == "text-embedding-test"
    assert "dimensions" not in encoder.payloads[0][1]


def test_api_text_encoder_sends_dimensions_only_when_requested():
    encoder = FakeAPITextEncoder("text-embedding-test", request_dimensions=2)
    encoder.encode(["alpha"])

    assert encoder.payloads[0][1]["dimensions"] == 2


def test_api_text_encoder_reuses_cache(tmp_path):
    first = FakeAPITextEncoder("text-embedding-test", cache_dir=tmp_path)
    first.encode(["cached"])

    second = FakeAPITextEncoder("text-embedding-test", cache_dir=tmp_path)
    values = second.encode(["cached"])

    np.testing.assert_array_equal(values, np.asarray([[0.0, 6.0]], dtype=np.float32))
    assert second.calls == 0


def test_api_encoder_cli_keeps_model_config_out_of_arguments():
    args = build_parser().parse_args([
        "preprocess",
        "--raw-dir", "raw",
        "--output-dir", "processed",
        "--encoder", "api",
    ])

    assert args.encoder == "api"
    assert not hasattr(args, "api_base_url")
    assert not hasattr(args, "api_output_dim")


def test_api_text_encoder_loads_configuration_from_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    cache_dir = tmp_path / "cache"
    env_path.write_text(
        "\n".join([
            "DYSEMKT_API_MODEL=text-embedding-test",
            "DYSEMKT_API_BASE_URL=https://example.test/v1",
            "DYSEMKT_API_KEY_ENV=EXAMPLE_API_KEY",
            "DYSEMKT_API_OUTPUT_DIM=2",
            "DYSEMKT_API_REQUEST_DIMENSIONS=2",
            f"DYSEMKT_API_CACHE_DIR={cache_dir}",
            "DYSEMKT_API_TIMEOUT=12.5",
            "DYSEMKT_API_MAX_RETRIES=4",
        ]),
        encoding="utf-8",
    )
    for name in (
        "DYSEMKT_API_MODEL",
        "DYSEMKT_API_BASE_URL",
        "DYSEMKT_API_KEY_ENV",
        "DYSEMKT_API_OUTPUT_DIM",
        "DYSEMKT_API_REQUEST_DIMENSIONS",
        "DYSEMKT_API_CACHE_DIR",
        "DYSEMKT_API_TIMEOUT",
        "DYSEMKT_API_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)

    load_env_file(env_path)
    encoder = APITextEncoder.from_env(batch_size=3)

    assert encoder.model_name == "text-embedding-test"
    assert encoder.base_url == "https://example.test/v1"
    assert encoder.api_key_env == "EXAMPLE_API_KEY"
    assert encoder.dimension == 2
    assert encoder.request_dimensions == 2
    assert encoder.batch_size == 3
    assert encoder.timeout == 12.5
    assert encoder.max_retries == 4
    assert encoder.cache_dir == cache_dir
