from typing import Literal
from taranis_base_bot.config import CommonSettings


class Settings(CommonSettings):
    MODEL: Literal["criticality_classifier"] = "criticality_classifier"
    PACKAGE_NAME: str = "cybersec_priority_bot"
    HF_MODEL_INFO: bool = False
    PAYLOAD_SCHEMA: dict[str, dict] = {
        "story_id": {"type": "str", "required": True},
    }


Config = Settings()
