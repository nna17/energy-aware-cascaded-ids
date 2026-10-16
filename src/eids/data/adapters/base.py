"""Every dataset is read through one adapter, so the rest of the code never sees dataset specifics."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd

from eids.utils.config import Config

FINE_LABEL = "label_fine"
BENIGN_CATEGORY = "Benign"
META_PREFIX = "meta_"            # provenance columns (device, time) used only for splitting, never as features
META_DEVICE = "meta_device"
META_TIME = "meta_time"          # seconds since the epoch at the start of the record's window


class DatasetAdapter(ABC):
    name: str
    benign_label: str  # fine-grained name of benign traffic

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Return feature columns plus a `label_fine` column with the original class names."""

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Same content as `load_raw`, in pieces. Multi-file datasets override this to bound memory."""
        yield self.load_raw()

    @abstractmethod
    def leaky_columns(self) -> list[str]:
        """Columns that identify the capture rather than the behaviour (IPs, ports, timestamps, payloads)."""

    @abstractmethod
    def category_of(self, fine_label: str) -> str:
        """Map a fine-grained class to its attack category (benign maps to BENIGN_CATEGORY)."""
