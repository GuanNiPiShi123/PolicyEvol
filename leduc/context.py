# -*- coding: utf-8 -*-

from typing import List

from rich.console import Console
from rich.prompt import Prompt

from setting import Settings
from agent import PEAgent

class Context:
    def __init__(self, console: Console, settings: Settings, webcontext=None) -> None:
        #self.clock: int = 0
        self.console: Console = console
        #self.user_agent: PEAgent = None
        self.robot_agents: List[PEAgent] = []
        #self.observations = ["Beginning of the day, people are living their lives."]
        #self.timewindow_size = 3
        #self.observations_size_history = []
        self.settings = settings
        self.webcontext = webcontext

    def print(self, message: str, style: str = None):
        if style:
            self.console.print(message, style=style)
        else:
            self.console.print(message)

        if self.webcontext:
            self.webcontext.send_response(message)

    def ask(self, message: str = "", choices: List[str] = None) -> str:
        if self.webcontext:
            return self.webcontext.ask_human(message, choices)
        else:
            if choices:
                return Prompt.ask(message, choices=choices, default=choices[0])
            else:
                return Prompt.ask(message)