from enum import Enum


class LookSide(str, Enum):
    """Look side of the platform."""

    LEFT = "left"
    RIGHT = "right"

    def __int__(self) -> int:
        return 1 if self is LookSide.LEFT else -1

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
            if value == cls.LEFT.value:
                return cls.LEFT
            if value == cls.RIGHT.value:
                return cls.RIGHT
        elif value == 1:
            return cls.LEFT
        elif value == -1:
            return cls.RIGHT
        return None
