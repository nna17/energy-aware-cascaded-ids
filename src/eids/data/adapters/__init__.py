from eids.data.adapters.base import DatasetAdapter
from eids.data.adapters.ciciiot2025 import CICIIoT2025Adapter
from eids.data.adapters.ciciot2023 import CICIoT2023Adapter
from eids.data.adapters.edge_iiotset import EdgeIIoTsetAdapter
from eids.data.adapters.synthetic import SyntheticAdapter
from eids.utils.config import Config

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    a.name: a for a in (CICIoT2023Adapter, CICIIoT2025Adapter, EdgeIIoTsetAdapter, SyntheticAdapter)
}


def get_adapter(cfg: Config) -> DatasetAdapter:
    try:
        return ADAPTERS[cfg.dataset](cfg)
    except KeyError:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}; known: {sorted(ADAPTERS)}") from None
