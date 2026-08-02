# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reinforcement learning (PPO via stable-baselines3) pipeline that trains an agent to fight the Sekiro boss "Genichiro" (or similar). The pipeline: `StateReader → RewardCalculator → SekiroEnv (Gymnasium) → PPO Agent → InputController → RestartManager → EpisodeLogger`.

Stages 1-7 (everything except real screen capture) run entirely on mock data — no game window needed. Stage 8 (real game integration via screen pixel reading) is scaffolded but **not implemented**: `PixelStateReader.read()` raises `NotImplementedError` even with full calibration. See `docs/live_game.md` for exact current state before assuming `--live` works.

Full docs are in `docs/` — `architecture.md` (module boundaries, data flow, action space), `configuration.md` (config.yaml fields), `training.md`, `testing.md`, `live_game.md`, `risks.md`, `roadmap.md`. Read the relevant doc before making non-trivial changes; this file only summarizes.

## Commands

```powershell
# activate venv first
.\venv\Scripts\Activate.ps1

# verify environment (all mock, no game needed)
python tests/test_state_reader.py
python tests/test_controller.py --dry-run
python tests/test_restart.py
python tests/test_reward.py
python tests/test_env.py
python tests/test_random_agent.py

# run a single test with its own args, e.g.
python tests/test_reward.py
python tests/test_env.py --skip-check-env --episodes 3 --steps 50

# train (mock)
python train.py --total-timesteps 1000 --run-name my_run

# resume training from checkpoint
python train.py --resume-from models/my_run.zip --run-name my_run_continued --total-timesteps 100000

# inference / watch agent
python play.py --model models/my_run.zip --episodes 10 --deterministic

# tensorboard
tensorboard --logdir logs/tensorboard
```

There is no lint/format config in the repo — don't assume one.

Tests are plain scripts with argparse, not pytest — run each file directly (`python tests/test_X.py`), not via a test runner. Each writes detailed output to `logs/<module>.log` in addition to stdout.

## Architecture

`sekiro_ai/` is organized as independent subpackages that only interact through shared data structures (`state_reader/schema.py`'s state dict, and the `Action` enum in `controller/action_map.py`) — they do not import each other's internals.

- `state_reader/` — `base.py` defines the `StateReader` interface (`read() -> dict`). `mock_reader.py` (scripted or random mock data) and `pixel_reader.py` (real screen capture via mss/OpenCV, unimplemented) both implement it. `SekiroEnv` only depends on the interface, so swapping mock for real requires no env changes.
- `controller/` — `action_map.py` defines the 7-action `Action` enum (wait/attack/guard/parry/dodge_left/dodge_right/backstep) and maps it to keymap config. `input_controller.py` executes via `pydirectinput`, supports `dry_run` mode.
- `restart/restart_manager.py` — detects `player_dead`/`boss_dead` from state and runs the auto-restart key sequence via `InputController` (no duplicate input logic).
- `reward/reward_calculator.py` — `compute(prev_state, curr_state)` turns state deltas into a scalar reward using weights from config.
- `env/sekiro_env.py` — Gymnasium `Env` wrapper composing the above; `Box` observation space, `Discrete(7)` action space.
- `logging/episode_logger.py` — SB3 `BaseCallback` (`EpisodeCsvLogger`) appending one row per episode to `logs/episodes/<run-name>.csv`.
- `utils/config_loader.py` — loads and `lru_cache`s `config/config.yaml`; changes to the YAML require restarting the process to take effect.

Unified state dict (`state_reader/schema.py`): `player_hp`, `player_posture`, `boss_hp`, `boss_posture`, `player_pos`, `boss_pos`, `distance`, `boss_action`, `player_hit`, `can_parry`, `player_dead`, `boss_dead`.

### Key design constraints

- Dependencies used only for real-game mode (`mss`, `opencv-python`, `pygetwindow`, `pywin32`, `pydirectinput`) are imported lazily inside `PixelStateReader` / non-dry-run `InputController`, not at module top level — mock/dry-run mode works even if those packages aren't installed.
- `config/config.yaml`'s `controller.keymap` keys must exactly match the 7 lowercase action names; a typo is silently ignored (falls back to default) rather than erroring — always re-run `tests/test_controller.py --dry-run` after editing keymaps.
- `calibration` section in config is the only section that can't be prefilled — it requires manual pixel measurement against a running game window at a fixed resolution. `missing_calibration()` only checks the four bar-rect fields, not `resolution`/`hp_color_hsv_range`/`perilous_icon_template`.
- Reward weight signs matter: `player_hp_delta` and `boss_hp_delta` are both positive weights but represent opposite-signed effects in the formula (boss dropping HP is good, player dropping HP is bad). After editing `reward.weights`, run `tests/test_reward.py` before training to catch sign regressions in the 10 hand-built test cases.
- Windows-only for real-game features (`pydirectinput`, `pywin32`); mock mode is OS-agnostic but only tested on Windows.
