# -*- coding: utf-8 -*-

import json
import os
from typing import Dict,Type
from pathlib import Path
import logging
from pythonjsonlogger import jsonlogger

from langchain import chat_models, embeddings, llms
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLanguageModel

from langchain_community.llms import Tongyi

from setting import EmbeddingSettings, LLMSettings

from setting import Settings
#from agent_model import load_llm_from_config_1
#from agent_model import load_embedding_from_config_1

def verify_model_initialization(settings: Settings) -> str:
    try:
        load_llm_from_config_1(settings.model.llm)
    except Exception as e:
        return f"LLM initialization check failed: {e}"
    '''
    try:
        load_embedding_from_config_1(settings.model.embedding)
    except Exception as e:
        return f"Embedding initialization check failed: {e}"
    '''
    return "OK"


def load_json(filepath: Path) -> Dict:
    if not Path(filepath).exists():
        return {}
    with open(filepath, "r") as file:
        try:
            json_obj = json.load(file)
            return json_obj
        except json.JSONDecodeError as e:
            if os.stat(filepath).st_size == 0:
                # Empty file
                return {}
            else:
                raise e

def get_logging(logger_name,content=''):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logHandlerJson = logging.FileHandler('./log/'+logger_name+'.json')
        formatter = jsonlogger.JsonFormatter()
        logHandlerJson.setFormatter(formatter)
        logger.addHandler(logHandlerJson)

    logger.info(content)#content may be a dict like {"1_0":{"raw_obs":observation decription after taking ction}} or {"1_0":{"act": action has been taken, "talk_sentence":""}}


llm_type_to_cls_dict: Dict[str, Type[BaseLanguageModel]] = {
    "chatopenai": chat_models.ChatOpenAI,
    "openai": llms.OpenAI,
    "Qwen":Tongyi,
    "Llama":chat_models.ChatOpenAI
}

# ------------------------- Embedding models registry ------------------------ #
embedding_type_to_cls_dict: Dict[str, Type[Embeddings]] = {
    "openaiembeddings": embeddings.OpenAIEmbeddings
}

# ---------------------------------------------------------------------------- #
#                                LLM/Chat models                               #
# ---------------------------------------------------------------------------- #
def load_llm_from_config_1(config: LLMSettings) -> BaseLanguageModel:
    """Load LLM from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")

    if config_type not in llm_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")
        
    if config_type == "Qwen":
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv())
        DASHSCOPE_API_KEY=os.environ["DASHSCOPE_API_KEY"]

    cls = llm_type_to_cls_dict[config_type]
    return cls(**config_dict)

# ---------------------------------------------------------------------------- #
#                               Embeddings models                              #
# ---------------------------------------------------------------------------- #
def load_embedding_from_config_1(config: EmbeddingSettings) -> Embeddings:
    """Load Embedding from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")
    print(config)
    if config_type not in embedding_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} Embedding not supported")

    cls = embedding_type_to_cls_dict[config_type]
    return cls(**config_dict)

if __name__ == "__main__":
    agent_config = load_json(Path("./game_config/leduc_limit.json"))
    print(agent_config)