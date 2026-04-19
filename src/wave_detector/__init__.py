"""Hot strip wave detection package."""

from .config import DetectorConfig, load_config
from .pipeline import WaveDetector

__all__ = ["DetectorConfig", "load_config", "WaveDetector"]
