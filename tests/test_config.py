from src.config import load_config


def test_load_config():
    cfg = load_config("configs/config.yaml")
    assert cfg.top_k > 0
    assert cfg.chunk_size > cfg.chunk_overlap