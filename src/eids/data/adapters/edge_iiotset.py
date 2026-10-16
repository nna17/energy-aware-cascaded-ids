from __future__ import annotations

import pandas as pd

from eids.data.adapters.base import BENIGN_CATEGORY, FINE_LABEL, DatasetAdapter
from eids.utils.logging import get_logger

log = get_logger()

_CATEGORIES = {
    "DDoS_UDP": "DoS_DDoS",
    "DDoS_ICMP": "DoS_DDoS",
    "DDoS_TCP": "DoS_DDoS",
    "DDoS_HTTP": "DoS_DDoS",
    "Port_Scanning": "InformationGathering",
    "Fingerprinting": "InformationGathering",
    "Vulnerability_scanner": "InformationGathering",
    "MITM": "MITM",
    "SQL_injection": "Injection",
    "XSS": "Injection",
    "Uploading": "Injection",
    "Backdoor": "Malware",
    "Password": "Malware",
    "Ransomware": "Malware",
}


class EdgeIIoTsetAdapter(DatasetAdapter):
    """Edge-IIoTset: one CSV with packet/flow fields, `Attack_label` (binary) and `Attack_type` (fine)."""

    name = "edge_iiotset"
    benign_label = "Normal"

    def load_raw(self) -> pd.DataFrame:
        path = self.cfg.get_path("root") / self.cfg.get("file_name", "DNN-EdgeIIoT-dataset.csv")
        log.info(f"[{self.name}] reading {path}")
        df = pd.read_csv(path, low_memory=False)
        # Attack_label is the binary target: it must never be a feature.
        return df.drop(columns=["Attack_label"]).rename(columns={"Attack_type": FINE_LABEL})

    def leaky_columns(self) -> list[str]:
        # Hosts, timestamps, ports and raw payload/URI strings identify the capture, not the behaviour.
        return [
            "frame.time", "ip.src_host", "ip.dst_host",
            "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
            "http.file_data", "http.request.full_uri", "http.request.uri.query",
            "icmp.transmit_timestamp", "tcp.options", "tcp.payload",
            "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
        ]

    def category_of(self, fine_label: str) -> str:
        if fine_label == self.benign_label:
            return BENIGN_CATEGORY
        try:
            return _CATEGORIES[fine_label]
        except KeyError:
            raise ValueError(f"[{self.name}] unknown label {fine_label!r}; add it to the category map") from None
