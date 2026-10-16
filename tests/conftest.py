import pytest

from eids.data.datamodule import load_datamodule
from eids.utils.config import load_config


@pytest.fixture(scope="session")
def synthetic_cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    return load_config("configs/data/synthetic.yaml", processed_dir=str(out), n_samples=4000, n_features=16)


@pytest.fixture(scope="session")
def synthetic_dm(synthetic_cfg):
    return load_datamodule(synthetic_cfg, rebuild=True)
