"""sekiro_ai.state_reader.frame_stack.FrameStack 的独立测试脚本。

用法：
    python tests/test_frame_stack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.frame_stack import FrameStack
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("frame_stack", "frame_stack.log")


def make_frame(value: int) -> np.ndarray:
    """一个填充单一数值的 8x8 帧，这样堆叠后的切片可以通过标量填充值轻易区分。"""
    return np.full((8, 8), value, dtype=np.uint8)


def test_reset_fills_stack_with_first_frame() -> bool:
    fs = FrameStack(stack_size=4, frame_skip=4)
    stacked = fs.reset(make_frame(10))
    ok = stacked.shape == (8, 8, 4) and stacked.dtype == np.uint8 and np.all(stacked == 10)
    logger.info("[%s] reset fills all %d slots with first frame", "PASS" if ok else "FAIL", 4)
    return ok


def test_push_before_skip_interval_repeats_last_frame() -> bool:
    fs = FrameStack(stack_size=4, frame_skip=4)
    fs.reset(make_frame(1))
    # frame_skip=4：只有每第 4 次 push() 调用才会轮转进新帧。
    # reset 之后的第 1-3 次调用不应改变堆栈。
    before = fs.push(make_frame(99)).copy()
    stacked = fs.push(make_frame(50))
    ok = np.array_equal(before, stacked)
    logger.info("[%s] pushes before frame_skip interval don't change the stack", "PASS" if ok else "FAIL")
    return ok


def test_push_at_skip_interval_rotates_newest_frame() -> bool:
    fs = FrameStack(stack_size=4, frame_skip=2)
    fs.reset(make_frame(0))
    fs.push(make_frame(11))  # 第 1 次调用：还在跳过间隔内，不轮转
    stacked = fs.push(make_frame(22))  # 第 2 次调用：达到跳过间隔，轮转进 22
    ok = stacked[:, :, -1][0, 0] == 22
    logger.info("[%s] newest frame lands at last stack index after frame_skip calls", "PASS" if ok else "FAIL")
    return ok


def test_stack_order_oldest_to_newest() -> bool:
    fs = FrameStack(stack_size=3, frame_skip=1)
    fs.reset(make_frame(0))
    fs.push(make_frame(1))
    fs.push(make_frame(2))
    stacked = fs.push(make_frame(3))
    ok = (
        stacked[0, 0, 0] == 1
        and stacked[0, 0, 1] == 2
        and stacked[0, 0, 2] == 3
    )
    logger.info("[%s] stack orders oldest frame at index 0, newest at last index", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    tests = [
        test_reset_fills_stack_with_first_frame,
        test_push_before_skip_interval_repeats_last_frame,
        test_push_at_skip_interval_rotates_newest_frame,
        test_stack_order_oldest_to_newest,
    ]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d FrameStack tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d FrameStack tests PASSED.", len(tests))
    logger.info("Test finished. Log written to logs/frame_stack.log")


if __name__ == "__main__":
    main()
