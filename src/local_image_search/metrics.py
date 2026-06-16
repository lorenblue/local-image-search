from __future__ import annotations

import resource
import sys


def memory_status() -> dict:
    return {
        "currentMb": _bytes_to_mb(_current_rss_bytes()),
        "peakMb": _peak_rss_mb(),
    }


def format_memory_status() -> str:
    memory = memory_status()
    return f"mem {memory['currentMb']:.1f} MB current, {memory['peakMb']:.1f} MB peak"


def _current_rss_bytes() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_bytes = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return _bytes_to_mb(peak_bytes)


def _bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)
