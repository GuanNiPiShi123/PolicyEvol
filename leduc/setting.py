# -*- coding: utf-8 -*-

import os

from pydantic.v1 import BaseModel, BaseSettings, Extra
from typing import Dict, Type

class LLMSettings(BaseModel):
    """
    LLM/ChatModel related settings
    """

    type: str = "chatopenai"

    class Config:
        extra = Extra.allow

class EmbeddingSettings(BaseModel):
    """
    Embedding related settings
    """

    type: str = "openaiembeddings"

    class Config:
        extra = Extra.allow

class ModelSettings(BaseModel):
    """
    Model related settings
    """

    type: str = ""
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()

    class Config:
        extra = Extra.allow

class Settings(BaseSettings):
    """
    Root settings
    """

    name: str = "default"
    model: ModelSettings = ModelSettings()

    class Config:
        env_prefix = "PE-Agent_"
        env_file_encoding = "utf-8"
        extra = Extra.allow

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
            
        ):
            return (
                init_settings,
                #json_config_settings_source,
                env_settings,
                file_secret_settings,
            )



# ---------------------------------------------------------------------------- #
#                             Preset configurations                            #
# ---------------------------------------------------------------------------- #
class OpenAIGPT4Settings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-0613",openai_api_key="",max_tokens=3000, temperature=0.1,  request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings", openai_api_key="")

class OpenAIGPT432kSettings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-32k-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-32k-0613", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")


class OpenAIGPT3_5TurboSettings(ModelSettings):
    type = "openai-gpt-3.5-turbo"
    llm = LLMSettings(type="chatopenai", model="gpt-3.5-turbo-16k-0613",openai_api_key="",max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings",openai_api_key="")


class OpenAIGPT3_5TextDavinci003Settings(ModelSettings):
    type = "openai-gpt-3.5-text-davinci-003"
    llm = LLMSettings(type="openai", model_name="text-davinci-003", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class OpenAIQwenSettings(ModelSettings):
    # NOTE: Qwen is in waitlist
    type = "Qwen"
    llm = LLMSettings(type="Qwen", model_name="qwen-max", temperature=0.1,  request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class DeepSeekSettings(ModelSettings):
    # NOTE: Qwen is in waitlist
    type = "DeepSeek"
    llm = LLMSettings(type="DeepSeek", model_name="DeepSeek-R1", temperature=0.1,  request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class OpenAILlamaSettings(ModelSettings):#model =Llama-3.1-70B-Instruct
    # NOTE: Llama is in waitlist
    type = "Llama"
    llm = LLMSettings(type="Llama", model="gpt-4-0613",openai_api_key="",max_tokens=3000, temperature=0.1, request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings", openai_api_key="")

# ------------------------- Model settings registry ------------------------ #
model_setting_type_to_cls_dict: Dict[str, Type[ModelSettings]] = {
    "openai-gpt-4-0613": OpenAIGPT4Settings,
    "openai-gpt-4-32k-0613": OpenAIGPT432kSettings,
    "openai-gpt-3.5-turbo": OpenAIGPT3_5TurboSettings,
    "openai-gpt-3.5-text-davinci-003": OpenAIGPT3_5TextDavinci003Settings,
    "Qwen":OpenAIQwenSettings,
    "Llama":OpenAILlamaSettings,
    "DeepSeek":DeepSeekSettings
}

def load_model_setting(type: str) -> ModelSettings:
    if type not in model_setting_type_to_cls_dict:
        raise ValueError(f"Loading {type} setting not supported")

    cls = model_setting_type_to_cls_dict[type]
    return cls()