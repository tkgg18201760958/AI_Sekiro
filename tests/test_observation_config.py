"""sekiro_ai.state_reader.observation_config.ObservationConfig 的独立测试脚本。

用法：
    python tests/test_observation_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.observation_config import ObservationConfig
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("observation_config", "observation_config.log")


def test_defaults_with_no_config() -> bool:
    cfg = ObservationConfig.from_config({})
    ok = cfg.frame_size == (84, 84) and cfg.frame_skip == 4 and cfg.stack_size == 4
    logger.info("[%s] defaults with empty config: %s", "PASS" if ok else "FAIL", cfg)
    return ok


def test_overrides_from_config() -> bool:
    cfg = ObservationConfig.from_config(
        {"observation": {"frame_size": [42, 42], "frame_skip": 2, "stack_size": 3}}
    )
    ok = cfg.frame_size == (42, 42) and cfg.frame_skip == 2 and cfg.stack_size == 3
    logger.info("[%s] overrides from config.yaml's observation section: %s", "PASS" if ok else "FAIL", cfg)
    return ok


def test_partial_override_keeps_other_defaults() -> bool:
    cfg = ObservationConfig.from_config({"observation": {"frame_skip": 8}})
    ok = cfg.frame_size == (84, 84) and cfg.frame_skip == 8 and cfg.stack_size == 4
    logger.info("[%s] partial override keeps unspecified fields at default: %s", "PASS" if ok else "FAIL", cfg)
    return ok


def main() -> None:
    tests = [test_defaults_with_no_config, test_overrides_from_config, test_partial_override_keeps_other_defaults]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d ObservationConfig tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d ObservationConfig tests PASSED.", len(tests))
    logger.info("Test finished. Log written to logs/observation_config.log")


if __name__ == "__main__":
    main()
