from __future__ import annotations

from .collector import collect_dataset, collect_instance_samples, main

__all__ = ["collect_dataset", "collect_instance_samples", "main"]


if __name__ == "__main__":
    main()
