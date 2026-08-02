"""MockStateReader 的 read_frame() 合成图像生成功能的独立测试脚本。

用法：
    python tests/test_mock_frame.py
    python tests/test_mock_frame.py --save-frames 5   # 同时导出 PNG 供肉眼检查
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.mock_reader import MockStateReader
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("mock_frame", "mock_frame.log")

FRAME_PREVIEW_DIR = Path(__file__).resolve().parent.parent / "logs" / "frame_preview"


def test_read_frame_shape_and_dtype() -> bool:
    reader = MockStateReader(mode="scripted", seed=0)
    frame = reader.read_frame()
    ok = frame.shape == (84, 84) and frame.dtype == np.uint8
    logger.info("[%s] read_frame() shape=%s dtype=%s (expected (84,84) uint8)", "PASS" if ok else "FAIL", frame.shape, frame.dtype)
    return ok


def test_full_hp_bar_brighter_than_empty_hp_bar() -> bool:
    """满血条应该比同一位置几乎空的血条有更多的亮（已填充）像素 ——
    这是"绘制出的血条确实跟随 GameState 数值变化"的一个代理验证方式，
    不需要精确到像素级的断言。"""
    reader = MockStateReader(mode="scripted", seed=0)
    reader._state.player_hp = 1.0
    full_frame = reader.read_frame()
    reader._state.player_hp = 0.05
    empty_frame = reader.read_frame()
    ok = full_frame.sum() > empty_frame.sum()
    logger.info(
        "[%s] full-HP frame brightness sum (%d) > near-empty-HP frame brightness sum (%d)",
        "PASS" if ok else "FAIL", int(full_frame.sum()), int(empty_frame.sum()),
    )
    return ok


def save_frames(n: int) -> None:
    if FRAME_PREVIEW_DIR.exists():
        shutil.rmtree(FRAME_PREVIEW_DIR)
    FRAME_PREVIEW_DIR.mkdir(parents=True)

    import cv2

    reader = MockStateReader(mode="scripted", seed=0)
    for i in range(n):
        reader.read()  # 推进脚本化轨迹
        frame = reader.read_frame()
        path = FRAME_PREVIEW_DIR / f"mock_frame_{i:03d}.png"
        cv2.imwrite(str(path), frame)
    logger.info("Saved %d mock frames to %s", n, FRAME_PREVIEW_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MockStateReader.read_frame().")
    parser.add_argument("--save-frames", type=int, default=0, help="Also save N preview PNGs to logs/frame_preview/.")
    args = parser.parse_args()

    tests = [test_read_frame_shape_and_dtype, test_full_hp_bar_brighter_than_empty_hp_bar]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d MockStateReader.read_frame() tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d MockStateReader.read_frame() tests PASSED.", len(tests))

    if args.save_frames > 0:
        save_frames(args.save_frames)

    logger.info("Test finished. Log written to logs/mock_frame.log")


if __name__ == "__main__":
    main()
