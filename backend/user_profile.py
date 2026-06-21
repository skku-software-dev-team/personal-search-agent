import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_PROFILE_PATH = Path(__file__).parent / "user_profile.json"


class UserProfile(BaseModel):
    goal: str | None = None
    fields: list[str] = []
    level: str | None = None
    timeline: str | None = None


def load_profile() -> UserProfile:
    if _PROFILE_PATH.exists():
        try:
            return UserProfile(**json.loads(_PROFILE_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return UserProfile()


def save_profile(profile: UserProfile) -> None:
    _PROFILE_PATH.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


@router.get("/user/profile", response_model=UserProfile)
async def get_profile():
    return load_profile()


@router.post("/user/profile", response_model=UserProfile)
async def set_profile(profile: UserProfile):
    save_profile(profile)
    return profile
