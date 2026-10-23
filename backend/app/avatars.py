"""Gender -> avatar, chosen automatically (the student never picks an image).

Gender is optional and used ONLY to choose the avatar. It is never used in matching or search.
The picture files live in static/avatars/ and are copied into the Xcode asset catalog with the
same names, so the app can also show them without a network call.
"""
from __future__ import annotations

Gender = str  # "female" | "male" | "other" | "unspecified"
GENDERS = ("female", "male", "other", "unspecified")

AVATAR_FILES = {
    "female": "avatar_female.png",
    "male": "avatar_male.png",
    "other": "avatar_neutral.png",
    "unspecified": "avatar_neutral.png",
}

_ALIASES = {
    "f": "female", "female": "female", "woman": "female", "girl": "female", "she": "female",
    "m": "male", "male": "male", "man": "male", "boy": "male", "he": "male",
    "o": "other", "other": "other", "non-binary": "other", "nonbinary": "other", "non binary": "other",
    "prefer not to say": "unspecified", "unspecified": "unspecified", "": "unspecified", "none": "unspecified",
}


def normalize_gender(raw) -> str:
    """Anything sensible -> one of GENDERS. Unknown or missing -> 'unspecified'."""
    return _ALIASES.get(str(raw or "").strip().lower(), "unspecified")


def avatar_file(gender) -> str:
    return AVATAR_FILES[normalize_gender(gender)]


def avatar_url(gender, base: str = "/static/avatars") -> str:
    return f"{base}/{avatar_file(gender)}"
