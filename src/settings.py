from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import SettingsConfigDict, BaseSettings
from pathlib import Path
from typing import Union


def get_root_dir():
    return Path(__file__).parent.parent


PATH_TO_ENV = get_root_dir() / "dev.env"


class OpenAISettings(BaseModel):
    url: Union[str, HttpUrl] = Field(
        default=HttpUrl("http://localhost"),
        description="Endpoint для LLM",
    )
    key: str = Field(
        default="EMPTY",
        description="api_key для LLM",
    )


class GenerationSettings(BaseModel):
    temperature: float = Field(
        default=1.0,
    )
    top_k: int = Field(
        default=60,
    )
    top_p: float = Field(
        default=0.8,
    )
    repetition_penalty: float = Field(
        default=1.0
    )
    frequency_penalty: float = Field(
        default=0.1,
    )
    min_p: float = Field(
        default=0.2,
    )


class ModuleSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PATH_TO_ENV.as_posix(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter='__',
        nested_model_default_partial_update=True,
        protected_namespaces=(),
    )

    openai_server: OpenAISettings = OpenAISettings()
    generation: GenerationSettings = GenerationSettings()
