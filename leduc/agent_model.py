# -*- coding: utf-8 -*-

from typing import Dict, List, Type

from rich.console import Console
from langchain import chat_models, embeddings, llms
#from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLanguageModel
from langchain.chat_models import init_chat_model


from langchain_community.llms import Tongyi

from agent import PEAgent
from setting import Settings
from setting import EmbeddingSettings, LLMSettings
from context import Context

def agi_init(
    agent_configs: List[dict],
    game_config:dict,
    console: Console,
    settings: Settings,
    webcontext=None,
) -> Context:
    ctx = Context(console, settings, webcontext)
    ctx.print("Creating all agents one by one...", style="yellow")
    for idx, agent_config in enumerate(agent_configs):
        agent_name = agent_config["name"]
        with ctx.console.status(f"[yellow]Creating agent {agent_name}..."):
            #player initialization
            agent = PEAgent(
                name=agent_config["name"],
                personality=agent_config["personality"],
                rule=game_config["game_rule"],
                game_name=game_config["name"],
                observation_rule=game_config["observation_rule"],
                llm=load_llm_from_config(ctx.settings.model.llm),
                reflection_threshold=8,
                belief = ""
            )

            if agent_config["memories"] !=[]:
                agent.memory=agent_config["memories"]

        ctx.robot_agents.append(agent)
        ctx.print(f"Agent {agent_name} successfully created", style="green")

    ctx.print("Policy Evolution Agent started...")

    return ctx

# ------------------------- LLM/Chat models registry ------------------------- #
llm_type_to_cls_dict: Dict[str, Type[BaseLanguageModel]] = {
    "chatopenai": chat_models.ChatOpenAI,
    "openai": llms.OpenAI,
    "Qwen":Tongyi,
    "Llama":chat_models.ChatOpenAI,
    "DeepSeek":init_chat_model
}

# ------------------------- Embedding models registry ------------------------ #
'''
embedding_type_to_cls_dict: Dict[str, Type[Embeddings]] = {
    "openaiembeddings": embeddings.OpenAIEmbeddings
}
'''

# ---------------------------------------------------------------------------- #
#                                LLM/Chat models                               #
# ---------------------------------------------------------------------------- #
def load_llm_from_config(config: LLMSettings) -> BaseLanguageModel:
    """Load LLM from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")

    if config_type not in llm_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")
    if config_type == "Qwen":
        import os
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv())
        DASHSCOPE_API_KEY=os.environ["DASHSCOPE_API_KEY"]        
        cls = llm_type_to_cls_dict[config_type]        
    elif config_type == "Llama":
        pass
        #return llm
    elif config_type == "DeepSeek":
        model = init_chat_model(
            model="deepseek-chat",
            api_key="",
            api_base="https://api.deepseek.com/",
            temperature=0.8,
            max_tokens=2048,
            model_provider="deepseek",
            )
        return model
    else:        
        cls = llm_type_to_cls_dict[config_type]
    return cls(**config_dict)
