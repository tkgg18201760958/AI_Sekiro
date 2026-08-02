"""PixelStateReader 血条填充比例提取数学计算的独立测试脚本，针对已经
提交到 GAME_PIC/ 目录下的两张静态截图运行（开发期间没有真实游戏窗口可用——
关于这个测试能验证什么、不能验证什么，参见
docs/superpowers/specs/2026-08-02-pixel-frame-observation-design.md
的范围说明）。

用法：
    python tests/test_pixel_bar_extraction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekiro_ai.state_reader.pixel_reader import bar_fill_ratio
from sekiro_ai.utils.logging_setup import get_logger

logger = get_logger("pixel_bar_extraction", "pixel_bar_extraction.log")

GAME_PIC_DIR = Path(__file__).resolve().parent.parent / "GAME_PIC"

HP_HSV_RANGE = ([0, 100, 90], [10, 255, 255])
POSTURE_HSV_RANGE = ([15, 80, 80], [35, 255, 255])

# 这些 rect 是从 config/config.yaml 的 calibration 部分复制过来的
# （对着 GAME_PIC/BOSS.png，一张 1280x720 的截图测量得到）。
PLAYER_HP_BAR = (70, 647, 184, 15)
PLAYER_POSTURE_BAR = (507, 617, 266, 14)
BOSS_HP_BAR = (71, 60, 279, 14)
BOSS_POSTURE_BAR = (580, 40, 119, 10)


def load_image(name: str):
    img = cv2.imread(str(GAME_PIC_DIR / name))
    if img is None:
        raise FileNotFoundError(f"Could not load {GAME_PIC_DIR / name}")
    return img


def test_boss_hp_bar_mostly_full_in_boss_screenshot() -> bool:
    img = load_image("BOSS.png")
    ratio = bar_fill_ratio(img, BOSS_HP_BAR, HP_HSV_RANGE)
    ok = ratio > 0.8
    logger.info("[%s] BOSS.png boss_hp_bar fill=%.3f (expected > 0.8, boss near full HP)", "PASS" if ok else "FAIL", ratio)
    return ok


def test_boss_hp_bar_near_empty_without_boss_engaged() -> bool:
    """WITHOUT_BOSS.png 里完全没有渲染 boss HP 条（没有 boss 交战）——
    该 rect 应该读出接近零的填充比例，因为那里根本没有红色的血条填充色。"""
    img = load_image("WITHOUT_BOSS.png")
    ratio = bar_fill_ratio(img, BOSS_HP_BAR, HP_HSV_RANGE)
    ok = ratio < 0.1
    logger.info("[%s] WITHOUT_BOSS.png boss_hp_bar fill=%.3f (expected < 0.1, no boss HUD present)", "PASS" if ok else "FAIL", ratio)
    return ok


def test_player_hp_bar_full_when_at_rest() -> bool:
    img = load_image("WITHOUT_BOSS.png")
    ratio = bar_fill_ratio(img, PLAYER_HP_BAR, HP_HSV_RANGE)
    ok = ratio > 0.8
    logger.info("[%s] WITHOUT_BOSS.png player_hp_bar fill=%.3f (expected > 0.8, player at rest/full HP)", "PASS" if ok else "FAIL", ratio)
    return ok


def test_posture_bars_empty_when_not_in_combat() -> bool:
    img = load_image("WITHOUT_BOSS.png")
    player_ratio = bar_fill_ratio(img, PLAYER_POSTURE_BAR, POSTURE_HSV_RANGE)
    ok = player_ratio < 0.1
    logger.info("[%s] WITHOUT_BOSS.png player_posture_bar fill=%.3f (expected < 0.1, not in combat)", "PASS" if ok else "FAIL", player_ratio)
    return ok


def test_boss_posture_bar_high_during_boss_fight() -> bool:
    img = load_image("BOSS.png")
    ratio = bar_fill_ratio(img, BOSS_POSTURE_BAR, POSTURE_HSV_RANGE)
    ok = ratio > 0.8
    logger.info("[%s] BOSS.png boss_posture_bar fill=%.3f (expected > 0.8, mid-fight)", "PASS" if ok else "FAIL", ratio)
    return ok


def main() -> None:
    tests = [
        test_boss_hp_bar_mostly_full_in_boss_screenshot,
        test_boss_hp_bar_near_empty_without_boss_engaged,
        test_player_hp_bar_full_when_at_rest,
        test_posture_bars_empty_when_not_in_combat,
        test_boss_posture_bar_high_during_boss_fight,
    ]
    failures = sum(0 if t() else 1 for t in tests)
    if failures:
        logger.error("%d/%d bar extraction tests FAILED.", failures, len(tests))
    else:
        logger.info("All %d bar extraction tests PASSED.", len(tests))
    logger.info("Test finished. Log written to logs/pixel_bar_extraction.log")


if __name__ == "__main__":
    main()
