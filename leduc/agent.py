# -*- coding: utf-8 -*-

import time
import util
from typing import Optional, List, Tuple

from pydantic.v1 import BaseModel
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel

class PEAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    personality: str
    
    game_name: str
    rule: str
    observation_rule: str

    """Current activities of the character."""
    llm: BaseLanguageModel

    """The retriever to fetch related memories."""
    verbose: bool = False

    """When the total 'importance' of memories exceeds the above threshold, stop to reflect."""
    plan:str=""
    
    belief:str
    self_belief:str=""
    opponent_belief:str=""
    
    self_pattern: List = []
    opponent_pattern: List = []

    memory: List = []
    read_observation: str = ""
    short_memory_summary: str = ""
    #order:str ="first_hand"

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_long_memory(self, memory_content: str) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        self.memory.append(memory_content)
        return  self.memory

    def planning_module(self, observation: str,  recipient_name:str,  belief: str =None, valid_action_list: List[str] = None, short_memory_summary:str = "", pattern:str = "", mode: str = "second_tom") -> str:
        """Make Plans and Evaluate Plans."""
        """Combining these two modules together to save costs"""
        #and figure out how many chips will be in pot when you take different actions 
        #and figure out how many chips will be in pot when you take different actions 
        prompt_2tom = PromptTemplate.from_template(
            "You are the objective player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} against {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            +'{pattern}\n'
            + " Your observation about the game status now is: {observation}\n"
            + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
            + '{belief}\n'
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent.\n"
            + " List potential {recipient_name}'s actions and Estimate Winning/Lose/Draw Rate for Each Plan: From the perspective of {recipient_name} , please infer what action {recipient_name} wiil take with probability (normalize to number 100% in total) would do when {recipient_name} holds different cards and then calculate the winning/lose/draw rates when {recipient_name} holds different cards step by step. At last, please calculate the overall winning/lose/draw rates for each plan step by step considering  {recipient_name}'s behaviour pattern. Output in a tree-structure: "
                + "Output: Plan 1:  If I execute plan1.  "
                          +"The winning/lose/draw rates when {recipient_name} holds card1: Based on {recipient_name}'s behaviour pattern, In the xx round, because {recipient_name} holds card1  (probability) and the combination with current public card (if release)  (based on my belief on {recipient_name}), and if he sees my action, {recipient_name} will do action1 (probability) ( I actually hold card and the public card (if reveal) is , he holds card1 and the public card (if reveal), considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose  step by step ), action2 (probability) (considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose step by step  ),.. (normalize to number 100% in total); \n   Overall (winning rate for his card1) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability)  "
                          +"The winning/lose/draw rates when {recipient_name} holds card2: Based on {recipient_name}'s behaviour pattern, In the xx round, because {recipient_name} holds card2  (probability) and the combination with current public card (if release)  (based on my belief on {recipient_name}) , and if he sees my action, he will do action1 (probability) (I actually hold card and the public card (if reveal) is , he holds card1 and the public card (if reveal), considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose  step by step ).. action2 (probability) (normalize to number 100% in total) (considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose step by step ),.. ;..... continue ....\n Overall (winning rate for his card2) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability) "
                          +"...\n"
                          +"Plan1 overall {initiator_name}'s Winning/Lose/Draw rates : the Winning rate (probability) for plan 1 is (winning rate for his card1) + (winning rate for his card2) + .. ; Lose rate (probability) for plan 1 : (lose rate for his card1) + (lose rate for his card2) + .. ; Draw Rate (probability) for plan 1  : (draw rate for his card1) + (draw rate for his card2) + ... ;  (normalize to number 100% in total) for plan1 \n"
                +"Plan 2: If I execute plan2, The winning/lose/draw rates when {recipient_name} holds card1: Based on {recipient_name}'s behaviour pattern, In the xx round, if {recipient_name} holds card1  (probability)  and the combination with current public card (if release),  .. (format is similar with before ) ... continue .."
                +"Plan 3: .. Coninue ... "
                + " The number of payoffs for each plan: Understanding your current observation,  each new plans, please infer the number of wininng/lose payoffs for each plan step by step, Output: Plan1: After the action, All chips are in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules step by step) . After the action,  All chips in the pot:  If lose , the lose payoff would be:  (Calculated by Lose Payoff Rules step by step). Plan2:  After the action, All chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules step by step):  After the action, All chips in the pot:  If lose , the lose chips would be:  (Calculated by Lose Payoff Rules step by step). If the number of my chips in pots have no change, please directly output them. \n"
            + " Estimate Expected Chips Gain for Each Plan : Understanding all the information and Estimate Winning/Lose/Draw Rate for Each Plan, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule) step by step (Note that you should first consider whether to add chips into pot while taking differnt actions, then calculate the ecpected chip gain).\n"
            + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement. \n "
        )
        
        prompt_1tom = PromptTemplate.from_template(
                "You are the player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} against {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            + " {pattern} \n"
            + " Your observation about the game status now is: {observation}\n"
            + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
            + ' {belief}\n'
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent.\n"
            + " List potential {initiator_name}'s actions and Estimate Winning/Lose/Draw Rate for Each Plan: please infer what action {recipient_name} wiil take with probability (normalize to number 100% in total) would do when {recipient_name} holds different cards and then calculate the winning/lose/draw rates when {recipient_name} holds different cards step by step. At last, please calculate the overall winning/lose/draw rates for each plan step by step considering  {recipient_name}'s behaviour pattern. Output in a tree-structure: "    
            + " Output: Based on {recipient_name}'s behaviour pattern and analysis on {recipient_name}'s cards, "
                +"Winning/lose/draw rates when {recipient_name} holds card1 in the xx round; if {recipient_name} holds card1  (probability) (based on my belief on {recipient_name}) with the public card  (if release), {recipient_name} will do action1 (probability) (infer I will win/draw/lose step by step (considering Single Game Win/Draw/Lose Rule and my factual card analysis with public card (if release), his card analysis with public card (if release) step by step ), action2 (probability) (infer I will win/draw/lose step by step  ),.. (normalize to number 100% in total);    Overall (winning rate for his card1) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability)  "
                          +"The winning/lose/draw rates when {recipient_name} holds card2 in the xx round; If {recipient_name} holds card2  (probability) (based on my belief on {recipient_name}) with the public card  (if release),  he will do action1 (probability) (infer I will win/draw/lose (considering Single Game Win/Draw/Lose Rule and my factual card analysis with current public card (if release), his card analysis with current public card (if release)) step by step ).. action2 (probability) (normalize to number 100% in total) (infer I will win/draw/lose step by step ),..  based on  {recipient_name}'s behaviour pattern;..... continue .... Overall (winning rate for his card2) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability) "
                          +"..."
                          +"Overall {initiator_name}'s Winning/Lose/Draw rates : Based on the above analysis, the Winning rate (probability) is (winning rate for his card1) + (winning rate for his card2) + .. ; Lose rate (probability): (lose rate for his card1) + (lose rate for his card2) + .. ; Draw Rate (probability): (draw rate for his card1) + (draw rate for his card2) + ... ;  (normalize to number 100% in total). \n"
            + " Potential believes about the number of winning and lose payoffs for each plan: Understanding the game rule, your current observation, previous actions summarization, each new plans, Winning Payoff Rule,  Lose Payoff Rule, please infer your several believes about  the number of chips in pots for each plan step by step, Output: Plan1: Chips in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules in the game rule) :  After the action,  If lose , the lose payoff would be: . Plan2:  Chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules in the game rule):  After the action, If lose , the lose chips would be: . If the number of my chips in pots have no change, please directly output them. "
            + " Estimate Expected Chips Gain for Each Plan: Understanding the game rule, plans, and your knowledge about the {game_name}, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule), explain what is the results if you do not select the plan, and explain why is this final Expected Chips Gain reasonablely step by step?  (Note that you should first consider whether to add chips into pot while taking differnt actions, then calculate the ecpected chip gain.)\n"
            + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement. \n\n "
        )

        prompt_notom = PromptTemplate.from_template(
            "You are the player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} against {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            + "  {pattern} \n"
            + " Your observation about the game status now is: {observation}\n"
            + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
            + ' {belief}\n'
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent."
               + " Estimate Winning/Lose/Draw Rate for Each Plan: Understanding the given information, and your knowledge about the {game_name}, please estimate the success rate of each step of each plan step by step and the overall average winning/lose/draw rate  (normalize to number 100% in total) of each plan/strategy for the current game step by step following the templete: If I do plan1, because I hold card, the public information (if release) and Single Game Win/Draw/Lose Rule, I will win or Lose or draw (probability);  ... continue  .... Overall win/draw/lose rate: Based on the analysis, I can do the weighted average step by step to get that the overall weighted average winning rate is (probability), average lose rate is (probability), draw rate is (probability) (normalize to number 100% in total)\n "
               + " Potential believes about the number of winning and lose payoffs for each plan: Understanding the game rule, your current observation, previous actions summarization, each new plans, Winning Payoff Rule,  Lose Payoff Rule, please infer your several believes about  the number of chips in pots for each plan step by step, Output: Plan1: Chips in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules in the game rule) :  After the action,  Chips in the pot:  If lose , the lose payoff would be: . Plan2:  Chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules in the game rule):  After the action, Chips in the pot:   If lose , the lose chips would be: . If the number of my chips in pots have no change, please directly output them. "
               +" Estimate Expected Chips Gain for Each Plan: Understanding the game rule, plans, and your knowledge about the {game_name}, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule)., explain what is the results if you do not select the plan, and explain why is this final Expected Chips Gain reasonablely step by step? "
            + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement. \n\n "
        )

        agent_summary_description = short_memory_summary

        belief = self.belief if belief is None else belief

        kwargs = dict(

            recent_observations=agent_summary_description,
            #last_plan=last_plan,
            belief=belief,
            initiator_name=self.name,
            pattern=pattern,
            recipient_name=recipient_name,
            observation=observation,
            rule=self.rule,
            game_name=self.game_name,
            valid_action_list=valid_action_list
        )
        print("Plan")
        if mode == "automatic":
            plan_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom, verbose=self.verbose)
            plan_2tom = plan_chain_2tom.run(**kwargs)

            plan_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom, verbose=self.verbose)
            plan_1tom = plan_chain_1tom.run(**kwargs)
            merged_paln = self.plan_evaluation(plan_2tom ,plan_1tom, observation, recipient_name, belief, short_memory_summary, pattern)
            
            return merged_paln.strip()
        
        elif mode == "second_tom":
            plan_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom, verbose=self.verbose)
            plan_2tom = plan_chain_2tom.run(**kwargs)
            return plan_2tom.strip()
        
        elif mode == "first_tom":
            plan_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom, verbose=self.verbose)
            plan_1tom = plan_chain_1tom.run(**kwargs)
            return plan_1tom.strip()
        
        else:
            plan_chain_notom = LLMChain(llm=self.llm, prompt=prompt_notom, verbose=self.verbose)
            plan_notom =plan_chain_notom.run(**kwargs)
            return plan_notom.strip()

    def plan_evaluation(self, plan_2tom:str, plan_1tom:str, observation:str,  recipient_name:str,  belief:str,  short_memory_summary:str, pattern:str):
        prompt = PromptTemplate.from_template(
            "You are the player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} with {recipient_name}.\n"
            + " The game rule is: {rule}\n"
            + " {pattern} \n"
            + " Your observation about the game status currently is: {observation}\n"
            + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
            + ' {belief}\n'
            + " Here are two plans provided:\n(1){plan_2tom}\n(2){plan_1tom}\n"
            + " Understanding all given information, judge which plan is more resonable, and only give your ultimate answer: (1) or (2)."
        )

        agent_summary_description = short_memory_summary

        kwargs = dict(

            recent_observations=agent_summary_description,
            #last_plan=last_plan,
            belief=belief,
            initiator_name=self.name,
            pattern=pattern,
            recipient_name=recipient_name,
            observation=observation,
            rule=self.rule,
            game_name=self.game_name,
            plan_1tom=plan_1tom,
            plan_2tom=plan_2tom
        )

        plan_evaluation_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        plan_evaluation_result = plan_evaluation_chain.run(**kwargs)
        if "1" in plan_evaluation_result:
            return plan_2tom
        else:
            return plan_1tom

    def get_self_belief(self, observation: str, recipient_name: str, short_memory_summary:str, pattern:str = "", mode: str = "second_tom", oppo_pattern:str = "", oppo_belief:str = "", last_k:int=5) -> str:
        """React to get self belief."""

        prompt_2tom = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about your own behaviour reflextion and improved strategy is: {pattern} \n"
                + " Your estimated judgement about the behaviour reflextion and improved strategy of {recipient_name} is: {oppo_pattern} \n"
                + " Your estimated belief about {recipient_name} is: {oppo_belief} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Your previous game memory including observations, actions, conversations with {recipient_name} and your reflection is: \n{long_memory}\n"
                + " Understanding the game rule, the cards you have, your observation,  progress summarization in the current game and previous game history, your estimated action reflection and improved strategy, and your knowledge about the {game_name}, can you do following things? \n"
                
                + " Analyze your cards: Understanding all given information and your knowledge about the {game_name}, please analysis what is your possible combination, advantages and disadvantages of your cards in the current round step by step.\n"
                
                + " Make your goals of the current round: in order to win more chips in the end, think what words you want to say and what action you take to bluff {recipient_name}.\n "
                + "Please don't respond too much irrelevant information."
            )

        prompt_1tom = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about your own behaviour reflextion and improved strategy is: {pattern} \n"
                + " Your estimated judgement about the behaviour reflextion and improved strategy of {recipient_name} is: {oppo_pattern} \n"
                + " Your estimated belief about {recipient_name} is: {oppo_belief} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Your previous game memory including observations, actions, conversations with {recipient_name} and your reflection is: \n{long_memory}\n"
                + " Understanding the game rule, the cards you have, your observation, progress summarization in the current game and previous game history, your estimated action reflection and improved strategy, and your knowledge about the {game_name}, can you do following things? \n"
               
                + " Analysis your cards: Understanding all given information, please analysis what is your possible combination, advantages and disadvantages of your cards in the current round step by step.\n"
                + " Make your goals of the current round: in order to win more chips in the end, think what words you want to say and what action you take to bluff {recipient_name}.\n "
                + "Please don't respond too much irrelevant information."
            )
        agent_summary_description = short_memory_summary

        long_memory = self.memory[-last_k:]
        if len(long_memory) ==0:
            long_memory_str = "you didn't get any game memory."
        else:
            long_memory_str = "\n\n".join([o for o in long_memory])

        kwargs = dict(
            #agent_summary_description=agent_summary_description,
            long_memory=long_memory_str,
            recent_observations=agent_summary_description,
            agent_name=self.name,
            pattern= pattern,
            recipient_name=recipient_name,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule,
            oppo_belief=oppo_belief,
            oppo_pattern = oppo_pattern
        )
        print(recipient_name)

        if mode == "second_tom":
            belief_prediction_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom)
            belief_2tom = belief_prediction_chain_2tom.run(**kwargs)
            return belief_2tom.strip()
        elif mode == "first_tom":
            belief_prediction_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom)
            belief_1tom = belief_prediction_chain_1tom.run(**kwargs)
            return belief_1tom.strip()
        else:
            belief_prediction_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom)
            belief_2tom = belief_prediction_chain_2tom.run(**kwargs)

            belief_prediction_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom)
            belief_1tom = belief_prediction_chain_1tom.run(**kwargs)
            
            merged_self_belief = self.belief_evaluation(belief_2tom.strip(), belief_1tom.strip(), recipient_name, agent_summary_description, observation, pattern, last_k)
            return merged_self_belief.strip()

    def get_opponent_belief(self, observation: str, recipient_name: str,short_memory_summary:str, pattern:str = "",mode: str = "second_tom", last_k:int=5) -> str:
        """React to get opponent belief."""

        prompt_2tom = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about the behaviour pattern and character of {recipient_name} is: {pattern} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Your previous game memory including observations, actions, conversations with {recipient_name} and your reflection is: \n{long_memory}\n"
                + " Understanding the game rule, the cards you have, your observation, progress summarization in the current game and previous game history, the estimated behaviour pattern and character of {recipient_name}, the potential guess pattern of {recipient_name} on you, and your knowledge about the {game_name}, can you do following things?\n "
                + " Belief on {recipient_name}'s cards: Understanding all given information, please infer the probabilities about the cards of {recipient_name}  (normalize to number 100% in total) objectively step by step."
                + " Output: {recipient_name} saw my history actions (or not) and then did action1 (probability) in the 1st round , ... continue..... Before this round, {recipient_name}  see my history actions (or not) and  did action1 (probability), because of {recipient_name}'s behaviour pattern and the match with the public card (if release), {recipient_name} tends to have card1 (probability), card2 (probability) ..continue.. (normalize to number 100% in total).\n"
                + " Analysis on {recipient_name}'s Cards: please analysis what is {recipient_name}'s best combination and advantages of {recipient_name}'s cards in the current round step by step.\n"
                + " Potential {recipient_name}'s current believes about your cards: Understanding all given information and your knowledge about the {game_name}, if you were {recipient_name} (he can only observe your actions but cannot see your cards), please infer the {recipient_name}'s believes about your cards with probability (normalize to number 100% in total) step by step. Output: {agent_name} did action1 (probability) (after I did action or not) in the 1st round, , ... continue...  {agent_name} did action1 (probability) (after I did action or not)  in the current round, from the perspective of {recipient_name}, {agent_name} tends to have card1 (probability), card2 (probability) ... (normalize to number 100% in total).\n"
                + " Guess potential {recipient_name}'s goals in the next round: Understanding all given information and your knowledge about the {game_name}, if you were {recipient_name} (you can only observe his actions but cannot see his cards), please infer the {recipient_name}'s goal (if possible) after {agent_name} take the action. Output: when I do action1 (probability), {recipient_name} will do ...continue...\n" 
                +  "Please don't respond too much irrelevant information."
            )

        prompt_1tom = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about the behaviour pattern and character of {recipient_name} is: {pattern} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Your previous game memory including observations, actions, conversations with {recipient_name} and your reflection is: \n{long_memory}\n"
                + " Understanding the game rule, the cards you have, your observation, progress summarization in the current game and previous game history, the estimated behaviour pattern and character of {recipient_name}, and your knowledge about the {game_name}, can you do following things? \n"
                + " Belief on {recipient_name}'s cards: please infer the probabilities about the cards of {recipient_name} (normalize to number 100% total) step by step. Templete: In the 1st round, {recipient_name} did action1 (probability),  ... continue... In the current round, {recipient_name} did action1 (probability), because of {recipient_name}'s behaviour pattern and the match with the current public card (if release), he tends to have card1 (probability), card2 (probability) (normalize to number 100% in total). \n"
                + " Analysis on {recipient_name}'s Cards: please analysis what is {recipient_name}'s best combination and advantages of {recipient_name}'s cards in the current round step by step.\n"
                + "Please don't respond too much irrelevant information."
            )

        agent_summary_description = short_memory_summary

        long_memory = self.memory[-last_k:]
        if len(long_memory) ==0:
            long_memory_str = "you didn't get any game memory."
        else:
            long_memory_str = "\n\n".join([o for o in long_memory])

        kwargs = dict(
            #agent_summary_description=agent_summary_description
            long_memory=long_memory_str,
            recent_observations=agent_summary_description,
            agent_name=self.name,
            pattern= pattern,
            recipient_name=recipient_name,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule

        )
        print(recipient_name)

        if mode == "second_tom":
            belief_prediction_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom)
            belief_2tom = belief_prediction_chain_2tom.run(**kwargs)
            return belief_2tom.strip()
        elif mode == "first_tom":
            belief_prediction_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom)
            belief_1tom = belief_prediction_chain_1tom.run(**kwargs)
            return belief_1tom.strip()
        else:
            belief_prediction_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom)
            belief_2tom = belief_prediction_chain_2tom.run(**kwargs)

            belief_prediction_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom)
            belief_1tom = belief_prediction_chain_1tom.run(**kwargs)
            
            merged_self_belief = self.belief_evaluation(belief_2tom.strip(),belief_1tom.strip(), recipient_name, agent_summary_description, observation, pattern, last_k)
            return merged_self_belief.strip()

    def belief_evaluation(self, belief_2tom: str, belief_1tom: str, recipient_name: str, agent_summary_description:str, observation:str, pattern:str, last_k:int) -> str:
        prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about the behaviour pattern or improved strategy is: {pattern} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Your previous game memory including observations, actions, conversations with {recipient_name} and your reflection is: \n{long_memory}\n"
                + " Here are two believes provided:\n(1){belief_2tom}\n(2){belief_1tom}\n"
                + " Based on your understanding of the game rule, previous game history, current observation, behaviour pattern as well as your knowledge about the {game_name}, "
                + " judge which belief is more resonable, and only give your ultimate answer: (1) or (2). "
                )

        long_memory = self.memory[-last_k:]


        if len(long_memory) ==0:
            long_memory_str = "you didn't get any game memory."
        else:
            long_memory_str = "\n\n".join([o for o in long_memory])

        kwargs = dict(
            pattern=pattern,
            long_memory = long_memory_str,
            agent_name=self.name,
            recipient_name=recipient_name,
            recent_observations=agent_summary_description,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule,
            belief_2tom=belief_2tom,
            belief_1tom=belief_1tom

        )

        belief_evaluation_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        evalution_result = belief_evaluation_chain.run(**kwargs)
        if "1" in evalution_result:
            return belief_2tom
        else:
            return belief_1tom

    #self pattern analysis and reflection(i.e. self policy evolution)
    def get_self_pattern(self, old_self_pattern, opponent_pattern, recipient_name: str, last_k:int=20, mode:str='second_tom', public_card=None) -> str:
        """React to get a self policy."""
        
        if public_card == None:
            prompt_2tom = PromptTemplate.from_template(
                    "You are the objective player behind a NPC character called {agent_name}, and you are playing {game_name} against {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                    + " Your previous behavioral strategy pattern is: {old_self_pattern}\n"
                    + " Your current behavioral pattern analysis and reasoning about your opponent {recipient_name} is: {opponent_pattern}\n"
                    + " Please understand the game rule, all previous game records, current game pattern of {recipient_name} and your previous behavioral strategy, can you do following things for future games?\n "
                    + " Reflection: Reflex which your actions are right or wrong in previous games (especially when the public card does not release), and think why you win or lose concrete chips step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions.)\n "
                    + " Strategy Improvement: Understanding the above information combined with your reflection, think about what strategies I can adopt to exploit the game pattern of {recipient_name}. By taking {recipient_name}'s guess on my game pattern without public card observed into account, revise previous strategies for winning {recipient_name} step by step." 
                    + " Output: When I hold card without public card, and see the action of the opponent, I would like to do action 1 (probabilities), action2 (probabilities) (normalize to number 100% in total), to my konwledge, I tend to act radically/conservatively/neutrally/flexibly, because I can infer that {recipient_name} is (or not) possibly bluffing; continue ... "
            )
        
        else:
            prompt_2tom = PromptTemplate.from_template(
                    "You are the objective player behind a NPC character called {agent_name}, and you are playing {game_name} against {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                    + " Your previous behavioral strategy pattern is: {old_self_pattern}\n"
                    + " Your current behavioral pattern analysis and reasoning about your opponent {recipient_name} is: {opponent_pattern}\n"
                    + " Please understand the game rule, all previous game records, current game pattern of {recipient_name} and your previous behavioral strategy, can you do following things for future games?\n "
                    + " Reflection: Reflex which your actions are right or wrong in previous games (especially in the public card release stage), and think why you win or lose concrete chips step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions.)\n "
                    + " Strategy Improvement: Understanding the above information combined with your reflection, think about what strategies I can adopt to exploit the game pattern of {recipient_name}. By taking {recipient_name}'s guess on my game pattern with public card observed into account, revise previous strategies for winning {recipient_name} step by step." 
                    + " Output: When I hold card and the public card, and see the action of the opponent, I would like to do action 1 (probabilities), action2 (probabilities) (normalize to number 100% in total), to my konwledge, I tend to act radically/conservatively/neutrally/flexibly, because I can infer that {recipient_name} is (or not) possibly bluffing; continue ... "
                )
           
        if public_card == None:
            prompt_1tom = PromptTemplate.from_template(
                    "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                    + " Your previous behavioral strategy adoption is: {old_self_pattern}\n"
                    + " Your current behavioral pattern analysis and reasoning about your opponent {recipient_name} is: {opponent_pattern}\n"
                    + " Please understand the game rule, all previous game records, current game pattern of {recipient_name} and your previous behavioral strategy, can you do following things for future games?\n "
                    + " Reflection: Reflex which your actions are right or wrong in previous games (especially when the public card does not release), and think why you win or lose concrete chips step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions.)\n "
                    + " Strategy Improvement: Understanding the above information combined with your reflection, think about what strategies you can adopt to exploit the game pattern of {recipient_name} for winning {recipient_name} in the whole game. Revise previous strategies for winning {recipient_name} step by step with the public card not observed. Output as a tree-structure." 
                    #+ " Output: When I hold card without public card, and see the action of the opponent, I would like to do action 1 (probabilities), action2 (probabilities) (normalize to number 100% in total), in this case, I tend to act radically/conservatively/neutrally/flexibly; continue ..."
                    )
        else: 
            prompt_1tom = PromptTemplate.from_template(
                    "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                    + " Your previous behavioral strategy adoption is: {old_self_pattern}\n"
                    + " Your current behavioral pattern analysis and reasoning about your opponent {recipient_name} is: {opponent_pattern}\n"
                    + " Please understand the game rule, all previous game records, current game pattern of {recipient_name} and your previous behavioral strategy, can you do following things for future games?\n "
                    + " Reflection: Reflex which your actions are right or wrong in previous games (especially in the public card release stage), and think why you win or lose concrete chips step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions.)\n "
                    + " Strategy Improvement: Understanding the above information combined with your reflection, think about what strategies you can adopt to exploit the game pattern of {recipient_name} for winning {recipient_name} in the whole game. Revise previous strategies for winning {recipient_name} step by step with the public card observed. Output as a tree-structure."                
                    #+ " Output: When I hold card with public card, and see the action of the opponent, I would like to do action 1 (probabilities), action2 (probabilities) (normalize to number 100% in total), in this case, I tend to act radically/conservatively/neutrally/flexibly; continue ... "
                    )

        prompt_notom = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                + " Your previous behavioral strategy adoption is: {old_self_pattern}\n"
                + " Your current behavioral pattern analysis and reasoning about your opponent {recipient_name} is: {opponent_pattern}\n"
                + " Your current behavioral pattern analysis about your opponent {recipient_name} is: {opponent_pattern}\n"
                + " Please first understand the game rule, all previous game records, current game pattern of {recipient_name} and your previous behavioral strategy. Then reflex which actions you took are right or wrong in previous games, think about what strategies do I need to adopt to win {recipient_name} for the whole game and revise or update previous strategies. And finally analysis I should behave radically/conservatively/neutrally/flexibly. Output your results. (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions)"                
                )

        long_memory = self.memory[-last_k:]

        if len(long_memory) == 0:
            long_memory_str = "You didn't get any game memory."
        else:
            long_memory_str = "\n\n".join([o for o in long_memory])

        if old_self_pattern == "":
            old_self_pattern = "You haven't got pattern about yourself yet."

        kwargs = dict(
            long_memory=long_memory_str,
            opponent_pattern = opponent_pattern,
            old_self_pattern =old_self_pattern,
            #game_pattern=game_pattern,
            agent_name=self.name,
            recipient_name=recipient_name,
            game_name=self.game_name,
            rule=self.rule
            #cshort_summarization = cshort_summarization
        )

        if mode == "automatic":
            reflection_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom, verbose=self.verbose)
            self_pattern_2tom = reflection_chain_2tom.run(**kwargs)

            reflection_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom, verbose=self.verbose)
            self_pattern_1tom = reflection_chain_1tom.run(**kwargs)
            
            merged_self_pattern = self.pattern_evaluation(self_pattern_2tom.strip(),self_pattern_1tom.strip(), recipient_name, long_memory_str)
            return merged_self_pattern.strip()
        
        elif mode == "second_tom":
            reflection_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom, verbose=self.verbose)
            self_pattern_2tom = reflection_chain_2tom.run(**kwargs)
            return self_pattern_2tom.strip()
        elif mode == "first_tom":
            reflection_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom, verbose=self.verbose)
            self_pattern_1tom = reflection_chain_1tom.run(**kwargs)
            return self_pattern_1tom.strip()
        else:
            reflection_chain_notom = LLMChain(llm=self.llm, prompt=prompt_notom, verbose=self.verbose)
            self_pattern_notom = reflection_chain_notom.run(**kwargs)
            return self_pattern_notom.strip()

    #opponent pattern analysis and reflection(i.e. opponent policy evolution)
    def get_opponent_pattern(self, old_opponent_pattern, recipient_name: str, last_k:int=20, mode:str='second_tom', public_card=None) -> str:
        """React to get a opponent policy."""
        
        if public_card == None:
            prompt_2tom = PromptTemplate.from_template(
                    "You are the objective player behind a NPC character called {agent_name}, and you are playing {game_name} with {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: {long_memory}\n"
                    + " Your previous behavioral pattern analysis about your opponent {recipient_name} is: {old_opponent_pattern}\n"
                    + " Please understand the game rule, all previous game summarization and previous game pattern of {recipient_name}, can you do following things for future games? \n"
                    + " Revise {recipient_name}'s game pattern judgement with public card not released: please infer, estimate and update all possible reasonable {recipient_name}'s game pattern/preferences for each card he holds with probability (normalize to number 100\% in total for each pattern item) when public card can't be observed, and also analyze how the {recipient_name}'s behaviour pattern/preferences are influenced by your actions, and output as a tree-structure output step by step."
                    + " Output: In the rounds with public card not released: when name holds card1: if {recipient_name} is the first to act, he would like to do action1 (probabilities), action2 (probabilities) ...; if {recipient_name} sees the action1/action2/action3 of the opponent or not, he would like to do action1 (probabilities), action2 (probabilities) ... when name holds card2, he would like ...(similar with before)\n "             
                    + " Acquire {recipient_name}'s guess on my game pattern with public card not released: Understanding all given information, please infer several reasonable believes about my game pattern/preference when holding different cards from the perspective of {recipient_name} (please consider the advantages of the card, actions) when public card does not release, output as a tree-structure output step by step."
                    + " Output: In the rounds with public card not released, when name holds card1, he would like to do action 1 (probabilities), action2 (probabilities)  (normalize to number 100% in total); when name holds card2, ...continue ... "
                    + " Judge {recipient_name}'s behavioral character: please infer {recipient_name}'s character based on {recipient_name}'s new generated game pattern and game history, and analysis how the {recipient_name}'s behaviour character is influenced by my actions with no public card released."
                    + " Output: To my konwledge, {recipient_name} tends to act radically/conservatively/neutrally/flexibly when I act xxx without public card; {recipient_name} tends to act ... when I when I act xxx without public card; continue ..."          
            )
        else:
            prompt_2tom = PromptTemplate.from_template(
                    "You are the objective player behind a NPC character called {agent_name}, and you are playing {game_name} with {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: {long_memory}\n"
                    + " Your previous behavioral pattern analysis about your opponent {recipient_name} is: {old_opponent_pattern}\n"
                    + " Please understand the game rule, all previous game summarization and previous game pattern of {recipient_name}, can you do following things for future games? \n"
                    + " Revise {recipient_name}'s game pattern judgement with public card released: Understanding all given information, please infer, estimate and update all possible reasonable {recipient_name}'s game pattern/preferences for each card he holds combined with public card with probability (normalize to number 100\% in total for each pattern item) when public card releases, and also analyze how the {recipient_name}'s behaviour pattern/preferences are influenced by yours actions, and output as a tree-structure output step by step."
                    + " Output: In the rounds with public card released, when name holds card1 and public card1 : if {recipient_name} is the first to act, he would like to do action1 (probabilities), action2 (probabilities) ...; if {recipient_name} sees the action1/action2/action3 of the opponent or not, he would like to do action1 (probabilities), action2 (probabilities) ... when name holds card2 and public card1: he would like ...(similar with before)\n" 
                    + " Acquire {recipient_name}'s guess on my game pattern: Understanding all given information, please infer several reasonable believes about my game pattern/preference when holding different cards from the perspective of {recipient_name} (please consider the advantages of the card, actions and the match with the public card) when public card releases, output as a tree-structure output step by step."
                    + " Output: In the rounds with public card released, when name holds card1 with public card1, he would like to do action 1 (probabilities), action2 (probabilities)  (normalize to number 100% in total); when name holds card2 with public card (if release), ...continue ... "       
                    + " Judge {recipient_name}'s behavioral character: please infer {recipient_name}'s character based on {recipient_name}'s new generated game pattern and game history, and analysis how the {recipient_name}'s behaviour character is influenced by my actions with different public card released."
                    + " Output: To my konwledge, {recipient_name} tends to act radically/conservatively/neutrally/flexibly when I act xxx  with public card1; {recipient_name} tends to act ... when I when I act xxx with public card1; continue ..."  
                    )

        
        if public_card == None:
            prompt_1tom = PromptTemplate.from_template(
                    "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                    + " Your previous behavioral pattern analysis about your opponent {recipient_name} is: {old_opponent_pattern}\n"
                    + " Please understand the game rule, all previous game summarization and previous game pattern of {recipient_name}, can you do following things for future games? \n"
                    + " Revise {recipient_name}'s game pattern with public card not released: please infer or update all possible reasonable {recipient_name}'s game pattern/preferences for each card he holds with probability (normalize to number 100\% in total for each pattern) as a tree-structure output step by step when public card can't be observed."
                    #+ " Output: In the rounds with public card not released, when name holds card1, he would like to do action (probabilities); when name holds card2, he would like to do action (probabilities), ... \n " 
                    + " Judge {recipient_name}'s character: please infer {recipient_name}'s behavioral character based on the newly-updated game pattern and game history. Output: To my konwledge, {recipient_name} tends to act radically/conservatively/neutrally/flexibly."
            )
        else:
            prompt_1tom = PromptTemplate.from_template(
                    "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                    + " The game rule is: {rule} \n"
                    #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                    + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                    + " Your previous behavioral pattern analysis about your opponent {recipient_name} is: {old_opponent_pattern}\n"
                    + " Please understand the game rule, all previous game summarization and previous game pattern of {recipient_name}, can you do following things for future games? \n"
                    +" Revise {recipient_name}'s game pattern with public card released:please infer or update all possible reasonable {recipient_name}'s game pattern/preferences for each card he holds combined with public cards with probability (normalize to number 100\% in total for each pattern item) as a tree-structure output step by step.  "
                    #+ " Output: In the rounds with public card released, when name holds card1 and public card1, he would like to do action (probabilities); when name holds card2 and public card1, he would like to do action (probabilities), ... \n " 
                    + " Judge {recipient_name}'s character: please infer {recipient_name}'s behavioral character based on the revised game pattern and game history. Output: To my konwledge, {recipient_name} tends to act radically/conservatively/neutrally/flexibly."
                    )
        

        prompt_notom = PromptTemplate.from_template(
                " You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                + " Your previous game memories including observations, actions, conversations with {recipient_name} and your reflection are: \n{long_memory}\n"
                + " Your previous behavioral pattern analysis about your opponent {recipient_name} is: {old_opponent_pattern}\n"
                + " Please first understand the game rule, all previous game records and previous game pattern of {recipient_name}. Then revise the game pattern analysis about {recipient_name} and output it. Finally, judge {recipient_name}'s charater (Output as: {recipient_name} tends to act radically/conservatively/neutrally/flexibly.) based on game history and the new generatd game pattern."
                )

        long_memory = self.memory[-last_k:]

        if len(long_memory) ==0:
            long_memory_str = "You didn't get any game memory."
        else:
            long_memory_str = "\n".join([o for o in long_memory])

        if old_opponent_pattern != "":
            old_opponent_pattern_str =  old_opponent_pattern
        else:
            old_opponent_pattern_str = "You haven't got opponnet pattern yet."

        kwargs = dict(
            long_memory=long_memory_str,
            old_opponent_pattern = old_opponent_pattern_str,
            #game_pattern=game_pattern,
            agent_name=self.name,
            recipient_name=recipient_name,
            game_name=self.game_name,
            rule=self.rule

        )

        if mode == "automatic":
            reflection_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom, verbose=self.verbose)
            opponent_pattern_2tom = reflection_chain_2tom.run(**kwargs)

            reflection_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom, verbose=self.verbose)
            opponent_pattern_1tom = reflection_chain_1tom.run(**kwargs)
            
            merged_opponent_pattern = self.pattern_evaluation(opponent_pattern_2tom.strip(), opponent_pattern_1tom.strip(), recipient_name, long_memory_str)
            return merged_opponent_pattern.strip()
        elif mode == "second_tom":
            reflection_chain_2tom = LLMChain(llm=self.llm, prompt=prompt_2tom, verbose=self.verbose)
            opponent_pattern_2tom = reflection_chain_2tom.run(**kwargs)
            return opponent_pattern_2tom.strip()
        elif mode == "first_tom":
            reflection_chain_1tom = LLMChain(llm=self.llm, prompt=prompt_1tom, verbose=self.verbose)
            opponent_pattern_1tom = reflection_chain_1tom.run(**kwargs)
            return opponent_pattern_1tom.strip()
        else:
            reflection_chain_notom = LLMChain(llm=self.llm, prompt=prompt_notom, verbose=self.verbose)
            opponent_pattern_notom = reflection_chain_notom.run(**kwargs)
            return opponent_pattern_notom.strip()

    def pattern_evaluation(self, pattern_2tom: str, pattern_1tom: str, recipient_name: str, long_memory_str: str) -> str:
        """React to select a better policy."""
        prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                #+ " Your previous round history in this current game is: {cshort_summarization}\n"
                + " Your previous game memories including observations, actions and conversations with {recipient_name} and your reflection are: {long_memory}\n"
                + " Here are two behavioral pattern analysis:\n(1){pattern_2tom}\n(2){pattern_1tom}\n"
                + " Based on your understanding the game rule, previous game history and your knowledge about the {game_name}, "
                + " choose the behavioral pattern analysis which is more precise and benificial for {agent_name}'s victory, and only give your ultimate answer directly: (1) or (2). "
                )
        kwargs = dict(
            long_memory=long_memory_str,
            agent_name=self.name,
            #cshort_summarization =cshort_summarization,
            recipient_name=recipient_name,
            game_name=self.game_name,
            rule=self.rule,
            pattern_2tom=pattern_2tom,
            pattern_1tom=pattern_1tom

        )
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        evalution_result = reflection_chain.run(**kwargs)
        if "1" in evalution_result:
            return pattern_2tom
        else:
            return pattern_1tom

    def get_summarization(self, recipient_name: str, game_memory: str, opponent_name:str, no_highsight_obs:bool) -> str:
        """Get a long memory summarization to save costs."""
        
        if no_highsight_obs:
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {opponent_name}. \n"
                + " The game rule is: {rule} \n"
                + " The observation conversion rules are: {observation_rule}\n"
                + " One game memory including observations, actions and conversations between {recipient_name} and {opponent_name} is: {long_memory}\n"
                + " Understanding the game rule, observation conversion rules and game history and your knowledge about the {game_name}, can you do following things:"
                + " History summarization: summary the game history with action, observation, and results information? Use the templete, and respond shortly: In the first round of game, name holds card1 does action .... continue ...\n"
                + "{opponent_name}'s card reasoning: because the card of {opponent_name} is not available, please infer {opponent_name}'s card with probability (100% in total) with your understanding about the above all information confidently step by step. Resonpse format can be like following: {recipient_name}'s  card is xx and public card (if release) is xxx, and {opponent_name} behaviours are xx, the current game result is xx, so {opponent_name}'s card may be xxx (with probability), xxx, ... \n"
                + " Reflection: Reflect which your actions or patterns are right or wrong in the finished game to win or lose conrete chips and briefly propose a reasonable strategy step by step  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions). "
                )
        
        else:
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {opponent_name}. \n"
                + " The game rule is: {rule} \n"
                + " The observation conversion rules are: {observation_rule}\n"
                + " One game memory including observations, actions and conversations between {recipient_name} and {opponent_name} is: {long_memory}\n"
                + " Understanding the game rule, observation conversion rules and game history and your knowledge about the {game_name}, can you do following things:"
                + " History summarization: summary the game history with action, observation, and results information? Use the templete, and respond shortly: In the first round of game, current public cards (if available) are ... name holds card1 does action ... continue ...\n"
                + " Reflection: Reflect which your actions or patterns are right or wrong in the finished game to win or lose conrete chips and briefly propose a reasonable strategy step by step  (Note that you have observed the cards and actions of the opponent at present). "
                )
            
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        kwargs = dict(
            observation_rule=self.observation_rule,
            long_memory=game_memory,
            agent_name=self.name,
            opponent_name=opponent_name,
            recipient_name=recipient_name,
            game_name=self.game_name,
            rule=self.rule
        )
        # print(kwargs)
        long_memory_summary = reflection_chain.run(**kwargs)
        long_memory_summary = long_memory_summary.strip()
        return long_memory_summary

    def get_short_memory_summary(self, observation: str, recipient_name: str, short_memory_summary:str) -> str:
        """React to get a current history."""
        prompt = PromptTemplate.from_template(
            "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}. \n"
            + " The game rule is: {rule}\n"
            + " Your current observation is: {observation}\n"
            + " The current game history including previous actions, observations and conversations is: {agent_summary_description}\n"
            + " Based on the game rule, your observation and your knowledge about the {game_name}, please summarize the current history and current state. Respond shortly: "
            + " In the first round, private card of name is ...(if available), name does action, and says ... continue ... Public card is ...(if available), name does action ..."
        )

        agent_summary_description = short_memory_summary

        kwargs = dict(
            agent_summary_description=agent_summary_description,
            #recent_observations=agent_summary_description,
            agent_name=self.name,
            recipient_name=recipient_name,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule,
            round=round
        )

        short_memory_summary_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.short_memory_summary = short_memory_summary_prediction_chain.run(**kwargs)
        self.short_memory_summary = self.short_memory_summary.strip()
        return self.short_memory_summary.strip()

    def convert_obs(self, observation: dict, recipient_name: str, user_index: str, valid_action_list:str) ->  str:
        """React to get a observation."""
        prompt = PromptTemplate.from_template(
            "You are the player behind a NPC character called {agent_name} attached to player index {user_index}, and you are playing the board game {game_name} against {recipient_name}. \n"
            + " The game rule is: {rule} \n"
            + " Your observation now is: {observation}\n"
            + " And the observation conversion rules are: {observation_rule}\n"
            + " You will receive a valid action list you can perform in this turn. \n"
            + " Your valid action list is: {valid_action_list}\n"
            + " Please convert {observation} and {valid_action_list} to the readable text based on the observation conversion rules and your knowledge about the {game_name} (respond shortly).\n\n"
            #+ " Response format can be like following: Now {agent_name}'s hand is xxx, and the community card is xxx (note it is none at round 1). The number of chips all players have invested is xxx. The actions {agent_name} can be choosen are {legal_actions}."
        )
        kwargs = dict(
            user_index=user_index,
            agent_name=self.name,
            rule=self.rule,
            recipient_name=recipient_name,
            observation=observation,
            valid_action_list=valid_action_list,
            game_name=self.game_name,
            observation_rule=self.observation_rule
        )
        obs_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.read_observation = obs_prediction_chain.run(**kwargs)
        self.read_observation = self.read_observation.strip()
        return self.read_observation

    def action_decision(self, valid_action_list: List[str], recipient_name:str, observation:str, short_memory_summary:str, belief, promp_head: str) -> Tuple[str,str]:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "You are the objective player behind a NPC character called {agent_name}, and you are playing the board game {game_name} against {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            + " Your observation about the game status now is: {observation}\n"
            + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
            + "Your belief about the goal of the current round is: {belief}\n"
            + "{plan}"
            + promp_head
            + "\n Based on your plan, please select the best action from the available action list: {valid_action_list} (Just one word) and say something to the opponent player to bluff or confuse him or keep silent to finally win the whole game and reduce the risk of your action (respond sentence only). Please respond them and split them by |."
        )

        agent_summary_description = short_memory_summary

        kwargs = dict(
            #agent_summary_description= agent_summary_description,
            # current_time=current_time_str,
            # relevant_memories=relevant_memories_str,
            agent_name= self.name,
            game_name=self.game_name,
            recipient_name=recipient_name,
            rule=self.rule,
            observation= observation,
            #agent_status= self.status,
            valid_action_list = valid_action_list,
            recent_observations = agent_summary_description,  
            plan = self.plan,
            belief = belief,
        )
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)

        result = action_prediction_chain.run(**kwargs)
        if "|" in result:
            result,result_comm = result.split("|",1)
        else:
            result_comm = ""
        return result.strip(),result_comm.strip()

    def make_act(self, observation: dict, opponent_name: str, player_index:int, valid_action_list: List, verbose_print:bool, game_idx:int, round:int, bot_short_memory:List, bot_long_memory:List, console, log_file_name='', mode='first_tom', stage="train", old_policy= "", no_highsight_obs=False):
                      #dict: how to convert to string format #self index       #List format                                               #actually it is step which starts from 0

        #LLM observation Interpreter: transfer to readable observation
        readable_text_amy_obs = self.convert_obs(observation, opponent_name, player_index, valid_action_list)
        #return：Private State(hand, legal actions)  && Public State(public card, chips)

        if verbose_print:
            console.print('readable_text_obs: ', style="red")
            print(readable_text_amy_obs)
        time.sleep(2.0)

        #get current history (summarize from observation and round memory)
        if len(bot_short_memory[player_index]) == 1:#at the first round
            short_memory_summary = f'{game_idx+1}th Game Starts. \n'.format(game_idx)+ readable_text_amy_obs
        else:# after several rounds
            short_memory_summary = self.get_short_memory_summary(observation=readable_text_amy_obs, recipient_name=opponent_name, short_memory_summary='\n'.join(bot_short_memory[player_index]))        
        #short_memory_summary='\n'.join(bot_short_memory[player_index])+"\nThe current state is:"+ readable_text_amy_obs
            
        if verbose_print:
            console.print('short_memory_summary: ', style="yellow")
            print(short_memory_summary)
        time.sleep(2.0)

        #get new opponent policy and self policy module
        if old_policy != "":
            old_policy_oppo = old_policy.split("|||")[0]
            old_policy_self = old_policy.split("|||")[1]
        else:
            old_policy_oppo = ""
            old_policy_self = ""
        
        #if stage == "train":
        if stage != "":            
            opponent_pattern = self.get_opponent_pattern(old_policy_oppo, opponent_name, mode=mode, public_card = observation["public_card"])
            time.sleep(1.0)
            self_pattern = self.get_self_pattern(old_policy_self, opponent_pattern, opponent_name,  mode=mode)

            console.print('new pattern: ', style="blue")
            new_policy = "|||".join([opponent_pattern, self_pattern])
            #print(new_policy)           
        else:
            opponent_pattern = old_policy_oppo
            self_pattern = old_policy_self
            new_policy = "|||".join([opponent_pattern, self_pattern])           
        time.sleep(2.0)

        #get belief module: opponent and self cunrrent information prediction(1st or 2nd ToM)
        if mode in ['second_tom', 'first_tom', 'automatic']:
            self.opponent_belief = self.get_opponent_belief(readable_text_amy_obs, opponent_name, short_memory_summary=short_memory_summary, pattern=opponent_pattern, mode = mode)
            time.sleep(1.0)
            self.self_belief = self.get_self_belief(readable_text_amy_obs, opponent_name, short_memory_summary=short_memory_summary, pattern=self_pattern, mode = mode, oppo_pattern = opponent_pattern, oppo_belief = self.opponent_belief)
            time.sleep(2.0)
            if verbose_print:
                console.print(self.name + " belief: " , style="deep_pink3")
                print(self.name + " belief: " + str(self.self_belief))

                console.print(opponent_name + " belief: " , style="deep_pink3")
                print(opponent_name + " belief: " + str(self.opponent_belief))
            belief= "Your belief about yourself is: "+ self.self_belief +"\nYour belief about "+opponent_name+" is:"+ self.opponent_belief+"\n"
        else:
            belief = ''
            self.self_belief = ""

        self.belief = belief
        pattern =  "Opponent "+opponent_name+"'s behavioral pattern and chareacter is:"+ opponent_pattern+"\nYour behavioral reflection and improve strategies are: "+ self_pattern
        time.sleep(2.0)

        #get plan and action module
        self.plan = self.planning_module(readable_text_amy_obs, opponent_name, belief = belief, valid_action_list=valid_action_list, short_memory_summary = short_memory_summary, pattern=pattern, mode=mode)
        if verbose_print:
            console.print(self.name + " plan: " , style="orchid")
            print(self.name + " plan: " + str(self.plan))

        time.sleep(2.0)
        promp_head = ''
        act, comm = self.action_decision(valid_action_list, opponent_name, readable_text_amy_obs, short_memory_summary, self.self_belief, promp_head)
        time.sleep(2.0)
        
        if log_file_name is not None:
            util.get_logging(logger_name=log_file_name + '_obs',
                        content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {"raw_obs": observation,
                                                                        "readable_text_obs": readable_text_amy_obs}})
            util.get_logging(logger_name=log_file_name + '_short_memory',
                        content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {
                            "raw_short_memory": '\n'.join(bot_short_memory[player_index]),
                            "short_memory_summary": short_memory_summary}})
            util.get_logging(logger_name=log_file_name + '_pattern',
                                content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): pattern})
            util.get_logging(logger_name=log_file_name + '_belief',
                            content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {"belief": belief}})
            util.get_logging(logger_name=log_file_name + '_plan',
                        content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {"plan": self.plan}})

        while act not in valid_action_list:
            print('Action + ', str(act), ' is not a valid action in valid_action_list, please try again.\n')
            promp_head += 'Action {act} is not a valid action in {valid_action_list}, please try again.\n'
            act, comm = self.action_decision(valid_action_list, opponent_name, readable_text_amy_obs, short_memory_summary, self.self_belief, promp_head)
        #print(self.name + " act: " + str(act))
        #print(comm)

        #update bot_short_memory and bot_long_memory
        bot_short_memory[player_index].append(f"{self.name} has the observation: {readable_text_amy_obs}, tries to take action: {act}, and says “{comm}” to {opponent_name}.")
        bot_short_memory[((player_index+1)%len(bot_short_memory))].append(f"The valid action list of {self.name} is {valid_action_list}, he tries to take action: {act}, and says “{comm}” to {opponent_name}.")

        bot_long_memory[player_index].append(
            f"{self.name} has the observation: {observation}, tries to take action: {act}, and says “{comm}” to {opponent_name}.")
        
        return act, comm, bot_short_memory, bot_long_memory, new_policy                                                                                                                                                                                                                          