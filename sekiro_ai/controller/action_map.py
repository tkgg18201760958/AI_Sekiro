"""Discrete Action enum and its mapping to concrete key/mouse input specs.

Per architecture.md section 5, the action space is a fixed Discrete(7). This
module owns two things:

1. The `Action` enum -- the only vocabulary the Gym env / PPO agent ever
   see (integers in, `Action` out). Nothing downstream should hardcode the
   raw int -> key relationship; that lives here and in config.yaml.
2. `load_keymap()` -- merges user overrides from config.yaml's
   `controller.keymap` section on top of `DEFAULT_KEYMAP`, so button
   remapping never requires touching code.

Each keymap entry is a small dict describing one of three input "shapes":
    {"type": "none"}                                   # no-op
    {"type": "mouse", "button": "left", "duration"?: s} # click or hold+release
    {"type": "key", "key": "e", "duration"?: s}         # tap or hold+release
    {"type": "combo", "keys": ["shift", "a"], "duration": s}  # hold N keys together
`duration` on mouse/key means "hold for this long then release" instead of a
plain tap -- used for e.g. parry (short right-click) and dodges (held
Shift+direction).
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

from ..utils.config_loader import load_config


class Action(IntEnum):
    WAIT = 0
    ATTACK = 1
    GUARD = 2
    PARRY = 3
    DODGE_LEFT = 4
    DODGE_RIGHT = 5
    BACKSTEP = 6


# Fallback mapping used when config.yaml has no controller.keymap section
# (or is missing entirely) -- keeps the module runnable/testable standalone.
DEFAULT_KEYMAP: dict[Action, dict[str, Any]] = {
    Action.WAIT: {"type": "none"},
    Action.ATTACK: {"type": "mouse", "button": "left"},
    Action.GUARD: {"type": "mouse", "button": "right"},
    Action.PARRY: {"type": "mouse", "button": "right", "duration": 0.08},
    Action.DODGE_LEFT: {"type": "combo", "keys": ["shift", "a"], "duration": 0.15},
    Action.DODGE_RIGHT: {"type": "combo", "keys": ["shift", "d"], "duration": 0.15},
    Action.BACKSTEP: {"type": "combo", "keys": ["shift", "s"], "duration": 0.15},
}


def load_keymap(config: dict[str, Any] | None = None) -> dict[Action, dict[str, Any]]:
    """Build the effective Action -> input-spec mapping.

    `config` is the already-loaded config.yaml dict (or a subset thereof for
    testing); pass None to load config.yaml from disk via config_loader.
    Unknown action names in config are ignored with no error, since a typo
    there should degrade to "use the default for that action", not crash.
    """
    cfg = config if config is not None else load_config()
    overrides = (cfg or {}).get("controller", {}).get("keymap", {})

    keymap = dict(DEFAULT_KEYMAP)
    for action in Action:
        spec = overrides.get(action.name.lower())
        if spec is not None:
            keymap[action] = spec
    return keymap
