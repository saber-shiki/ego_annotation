from __future__ import annotations

"""Stable vector layout for raw policy actions.

Coordinates are normalized but unbounded. Hard filtering is separate and never
changes the action retained by PPO for log-probability accounting.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


ACTION_LAYOUT_VERSION = "ego.annotation.action_layout.v2"


@dataclass(frozen=True)
class ActionBlock:
    name: str
    start: int
    stop: int
    kind: str
    slot: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ActionLayout:
    hand_slots: tuple[str, ...] = ("left", "right")
    object_slots: tuple[str, ...] = ("object:0",)
    contact_variables_per_hand: int = 1
    include_hand_scale: bool = False
    include_object_shared_scale: bool = False

    def __post_init__(self) -> None:
        if self.hand_slots != ("left", "right"):
            raise ValueError("fixed hand slots must be ('left', 'right')")
        if not self.object_slots:
            raise ValueError("at least one object action slot is required")
        if self.contact_variables_per_hand < 0:
            raise ValueError("contact variable count must be nonnegative")

    @property
    def version(self) -> str:
        return ACTION_LAYOUT_VERSION

    @property
    def blocks(self) -> tuple[ActionBlock, ...]:
        blocks: list[ActionBlock] = []
        offset = 0
        for slot, side in enumerate(self.hand_slots):
            for name, size in (("root", 3), ("articulation", 45), ("translation", 3)):
                blocks.append(ActionBlock(f"hand.{side}.{name}", offset, offset + size, f"hand_{name}", slot))
                offset += size
            if self.include_hand_scale:
                blocks.append(ActionBlock(f"hand.{side}.scale", offset, offset + 1, "hand_scale", slot))
                offset += 1
        for slot, object_name in enumerate(self.object_slots):
            for name, size in (("rotation", 3), ("translation", 3)):
                blocks.append(ActionBlock(f"object.{object_name}.{name}", offset, offset + size, f"object_{name}", slot))
                offset += size
        if self.include_object_shared_scale:
            blocks.append(ActionBlock("object.shared_scale", offset, offset + 1, "object_scale", 0))
            offset += 1
        for slot, side in enumerate(self.hand_slots):
            size = self.contact_variables_per_hand
            if size:
                blocks.append(ActionBlock(f"contact.{side}", offset, offset + size, "contact", slot))
                offset += size
        return tuple(blocks)

    @property
    def size(self) -> int:
        return self.blocks[-1].stop if self.blocks else 0

    def document(self) -> dict[str, object]:
        return {
            "schema": ACTION_LAYOUT_VERSION,
            "size": self.size,
            "hand_slots": list(self.hand_slots),
            "object_slots": list(self.object_slots),
            "contact_variables_per_hand": self.contact_variables_per_hand,
            "include_hand_scale": self.include_hand_scale,
            "include_object_shared_scale": self.include_object_shared_scale,
            "blocks": [
                {"name": block.name, "start": block.start, "stop": block.stop, "kind": block.kind, "slot": block.slot}
                for block in self.blocks
            ],
        }

    def block(self, name: str) -> ActionBlock:
        matches = [block for block in self.blocks if block.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def decode(self, vector: Sequence[float]) -> dict[str, np.ndarray]:
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (self.size,) or not np.isfinite(value).all():
            raise ValueError(f"action must be finite shape {(self.size,)}")
        return {block.name: value[block.slice].copy() for block in self.blocks}

    def presence_mask(self, hand_present: Sequence[bool], object_present: Sequence[bool],
                      contact_active: Sequence[bool] | None = None) -> np.ndarray:
        hand = np.asarray(hand_present, dtype=bool)
        objects = np.asarray(object_present, dtype=bool)
        contact = hand if contact_active is None else np.asarray(contact_active, dtype=bool)
        if hand.shape != (len(self.hand_slots),) or objects.shape != (len(self.object_slots),):
            raise ValueError("entity presence shape does not match action layout")
        if contact.shape != (len(self.hand_slots),):
            raise ValueError("contact mask shape does not match hand slots")
        mask = np.ones(self.size, dtype=bool)
        for block in self.blocks:
            if block.kind.startswith("hand_"):
                mask[block.slice] = hand[block.slot]
            elif block.kind.startswith("object_"):
                mask[block.slice] = objects[block.slot]
            else:
                mask[block.slice] = contact[block.slot]
        return mask
