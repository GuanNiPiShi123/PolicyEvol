# -*- coding: utf-8 -*-

import json
import os
import argparse
import random
import math 
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
import rlcard
from rlcard import models
from rlcard.utils import set_seed

from setting import Settings, load_model_setting
from agent_model import agi_init, load_llm_from_config
import util

from suspicion_agent import SuspicionAgent 

console = Console()
#advatage:(1) form lessons from past experiences (2)save time and guickly get access to opponent's character (3)tom:self and environment analysis

#TODO:
#(1)align human preference
#(2) retrieve best similar betting or game record

#LLama对应的交互包还需要改：在agent_model.py和util.py中的llm_type_to_cls_dict变量里

def get_agent_config(args, idx):    
    agent_config = {}
    while True:
        agent_file = args.player1_config if idx == 0 else args.player2_config
        try:
            agent_config = util.load_json(Path(agent_file))
            
            if agent_config == {}:
                console.print(
                    "Empty configuration, please provide a valid one", style="red"
                )
                continue
            
            agent_config["path"] = agent_file
            
            if args.load_memory:
                with open(args.memory_record,"r") as m_f:
                    agent_config["memories"] = json.load(m_f)["memories"]
            break
        
        except json.JSONDecodeError:
            console.print(
                "Invalid configuration, please provide a valid one", style="red"
            )
            agent_file = Prompt.ask(
                "Enter the path to the agent configuration file", default="./agent.json"
            )
            continue
    
    return agent_config
    #agent_config=[{"name": "board_game_expert","age": 27,"personality": "flexible","memories":[],"path":"./person_config/Persuader.json"},
    #{"name": "GoodGuy","age": 27,"personality": "flexible","memories":[],"path":"./person_config/GoodGuy.json"}]
    
def get_rule_model(rule_model_name, llm, opponent_name):
    
    if rule_model_name == 'cfr':
        rule_model = models.load('leduc-holdem-cfr').agents[0]
    elif rule_model_name == "suspicion-agent":
        game_config = util.load_json(Path("./suspicion_game_config/leduc_limit.json"))
        rule_model = SuspicionAgent(
                        name = opponent_name,
                        age=27,
                        rule=game_config["game_rule"],
                        game_name=game_config["name"],
                        observation_rule=game_config["observation_rule"],
                        status="N/A",  
                        llm=load_llm_from_config(llm),                   
                        reflection_threshold=8,
                    )
    else:
        import torch
        rule_model = torch.load(os.path.join('./models', 'leduc_holdem_'+rule_model_name+'_result/model.pth'), map_location='cpu', weights_only = False)
        rule_model.set_device('cpu')
    
    return rule_model  

def play_game(args, game_idx, ctx,  policy, rule_model, chips, stage, seed_number, log_file_name):
    
    private_policy = policy["private"]
    public_policy = policy["public"]
    
    bot_long_memory = []
    bot_short_memory = []
    #bot_short_memory[0] or bot_short_memory[1] records game_id, self obeservation and action as well as public card and opponents's action at each step, and win message
    #bot_long_memory[0] or bot_long_memory[1] records game_id, self obeservation, action or opponents's observation and action at each step
    
    oppo_bot_short_memory = []
    
    for i in range(args.agents_num):
        bot_short_memory.append([f'{game_idx+1}th Game Starts.'])
        bot_long_memory.append([f'{game_idx+1}th Game Starts.'])
        
        oppo_bot_short_memory.append([f'{game_idx+1}th Game Starts.'])
        
        
    
    if args.random_seed and stage == "train":
        env = rlcard.make('leduc-holdem', config={'seed': random.randint(0,10000)})
    else:
        env = rlcard.make('leduc-holdem', config={'seed': seed_number})
    env.reset()

    
    round = 0
    
    while not env.is_over():
        idx = env.get_player_id()# indicates which palyer should take action
        console.print("Player:" + str(idx))
        
        if round == 0:
            start_idx = idx#indicates which palyer should take action first in the round 0 
            
        if args.user_index == idx and args.user:#args.user_index means opponent index, args.user= true means that opponent uses baseline model such as cfr
            
            oppo_obs = env.get_state(env.get_player_id())['raw_obs']
                        
            if args.rule_model == "suspicion-agent":                    
                oppo_obs['game_num'] = game_idx+1
                oppo_obs['rest_chips'] = chips[args.user_index]
                oppo_obs['opponent_rest_chips'] = chips[(args.user_index + 1) % args.agents_num]
                console.print(oppo_obs, style="green")#opponent's observation before take action
                valid_action_list = env.get_state(env.get_player_id())['raw_legal_actions']

                my_agent_name  = ctx.robot_agents[(idx+1)%args.agents_num].name
                act, oppo_comm, oppo_bot_short_memory, _ = rule_model.make_act(oppo_obs, my_agent_name, idx, valid_action_list, verbose_print = args.verbose_print, game_idx = game_idx, round=round, bot_short_memory=oppo_bot_short_memory, bot_long_memory=bot_long_memory, console=console, log_file_name=None ,mode="first_tom")
                
                util.get_logging(logger_name = log_file_name + '_opponent_observation',
                                 content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {"raw_obs": oppo_obs}})
                #content={"1_0":{"raw_obs":observation decription after taking ction}}
                util.get_logging(logger_name = log_file_name + '_opponent_action',
                                 content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {
                                         "act": str(act), "talk_sentence": "“"+oppo_comm+"”"}})
                bot_short_memory[args.user_index].append(
                    f"{ctx.robot_agents[args.user_index].name} has the observation: {oppo_obs}, says “{oppo_comm}” to {ctx.robot_agents[(args.user_index+1)%args.agents_num].name}, and tries to take action: {act}.")
                bot_short_memory[(args.user_index + 1) % args.agents_num].append(
                    f"The valid action list of {ctx.robot_agents[args.user_index].name} is {env.get_state(env.get_player_id())['raw_legal_actions']}, he says “{oppo_comm}” to {ctx.robot_agents[(args.user_index+1)%args.agents_num].name}, and he tries to take action: {act}.")
                
                if args.no_hindsight_obs:
                    #do not add the opponent’s observation into the single game history after the end of each game
                    bot_long_memory[args.user_index].append(
                        f"{ctx.robot_agents[args.user_index].name} says “{oppo_comm}” to {ctx.robot_agents[(args.user_index+1)%args.agents_num].name}, and tries to take action: {act}.")
                else:
                    #add the opponent’s observation into the single game history after the end of each game
                    bot_long_memory[args.user_index].append(
                        f"{ctx.robot_agents[args.user_index].name} has the observation: {oppo_obs}, says “{oppo_comm}” to {ctx.robot_agents[(args.user_index+1)%args.agents_num].name}, and tries to take action: {act}.")
            
            else:
                console.print(oppo_obs, style="green")#opponent's observation before take action
                act,_ = rule_model.eval_step(env.get_state(env.get_player_id()))
                act = env._decode_action(act)

                util.get_logging(logger_name = log_file_name + '_opponent_observation',
                                 content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {"raw_obs": oppo_obs}})
                #content={"1_0":{"raw_obs":observation decription after taking ction}}
                util.get_logging(logger_name = log_file_name + '_opponent_action',
                                 content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {
                                         "act": str(act), "talk_sentence": str("")}})
                bot_short_memory[args.user_index].append(
                    f"{ctx.robot_agents[args.user_index].name} has the observation: {oppo_obs}, and tries to take action: {act}.")
                bot_short_memory[(args.user_index + 1) % args.agents_num].append(
                    f"The valid action list of {ctx.robot_agents[args.user_index].name} is {env.get_state(env.get_player_id())['raw_legal_actions']}, and he tries to take action: {act}.")
                if args.no_hindsight_obs:
                    #do not add the opponent’s observation into the single game history after the end of each game
                    bot_long_memory[args.user_index].append(
                        f"{ctx.robot_agents[args.user_index].name} tries to take action: {act}.")
                else:
                    #add the opponent’s observation into the single game history after the end of each game
                    bot_long_memory[args.user_index].append(
                        f"{ctx.robot_agents[args.user_index].name} has the observation: {oppo_obs}, and tries to take action: {act}.")
            console.print(act, style="green")            
                
        else:#it is trun of LLM agent to take action.
            amy = ctx.robot_agents[idx]# amy = PEAgent
            amy_obs = env.get_state(env.get_player_id())['raw_obs']
            amy_obs['game_num'] = game_idx+1
            amy_obs['rest_chips'] = chips[(args.user_index + 1) % args.agents_num]
            amy_obs['opponent_rest_chips'] = chips[args.user_index]
            console.print(amy_obs, style="green")#observation before take action
            
            valid_action_list = env.get_state(env.get_player_id())['raw_legal_actions']
            opponent_name = ctx.robot_agents[args.user].name

            if amy_obs["public_card"] == None:
                input_policy = private_policy
            else:
                input_policy = public_policy
                
            #core code: envoke LLM to give the commment, memory and action
            act, comm, bot_short_memory, bot_long_memory, new_policy = amy.make_act(amy_obs, opponent_name, idx, valid_action_list, verbose_print= args.verbose_print,
                                                                        game_idx = game_idx, round=round, bot_short_memory=bot_short_memory, bot_long_memory=bot_long_memory, console=console,
                                                                        log_file_name=log_file_name, mode=args.mode, stage = stage, old_policy = input_policy)
            util.get_logging(logger_name = log_file_name + '_PEAgent_observation',
                             content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {"raw_obs": amy_obs}})
            #content={"1_0":{"raw_obs":observation decription after taking ction}}
            util.get_logging(logger_name = log_file_name + '_PEAgent_action',
                             content={"GameID:"+str(game_idx + 1) + "_round:" + str(round+1): {
                                     "act": str(act), "talk_sentence": "“"+comm+"”"}})
            

            if amy_obs["public_card"] == None:
                private_policy = new_policy
            else:
                public_policy = new_policy
            
            console.print(act, style="green")
            
            oppo_bot_short_memory[args.user_index].append(f"The valid action list of {ctx.robot_agents[(args.user_index+1)%args.agents_num].name} is {valid_action_list}, he says “{comm}” to {ctx.robot_agents[args.user_index].name}, and he tries to take action: {act}.")
            oppo_bot_short_memory[(args.user_index+1) % args.agents_num].append(
                f"{ctx.robot_agents[(args.user_index+1)%args.agents_num].name} has the observation: {amy_obs}, says “{comm}” to {ctx.robot_agents[args.user_index].name}, and tries to take action: {act}.")

        env.step(act, raw_action=True)
        round += 1
    
    pay_offs = env.get_payoffs()
    print("Pay offs:")
    print(pay_offs)
    
    if pay_offs[0] > 0:
        win_message = f'{ctx.robot_agents[0].name} win {pay_offs[0]} chips, {ctx.robot_agents[1].name} lose {pay_offs[0]} chips.'
    else:
        win_message = f'{ctx.robot_agents[1].name} win {pay_offs[1]} chips, {ctx.robot_agents[0].name} lose {pay_offs[1]} chips.'
    #print(win_message)
    
    if len(bot_long_memory[start_idx]) == len(bot_long_memory[(start_idx+1)%args.agents_num]):
        long_memory = '\n'.join(
            [x + '\n' + y for x, y in zip(bot_long_memory[start_idx][1:], bot_long_memory[(start_idx+1)%args.agents_num][1:])])
    else: 
        long_memory = '\n'.join(
            [x + '\n' + y for x, y in zip(bot_long_memory[start_idx][1:-1], bot_long_memory[(start_idx+1)%args.agents_num][1:])])
        long_memory += bot_long_memory[start_idx][-1]
    
    long_memory = f'{game_idx+1}th Game Starts. ' + long_memory + win_message
    #print(long_memory)
    
    #get my agent score
    if (args.user_index+1) % args.agents_num == 0:
        score = pay_offs[0]
    else:
        score = pay_offs[1]
        
    #bot_short_memory: the first element is history list of position_0 user, the second element is history list of position_1 user
    #the short history list example of GoodGuy: ['1th Game Start', 'GoodGuy have the observation:xxx, and try to take action:xxx',
    #board_game_expert try to take action: xxx and says xxx to GoodGuy, ...(echange the turn)]
    
    #the short history list example of PEAgent: ['1th Game Start', 'The valid action list of GoodGuy is:raw_legal_actions,, and he tries to take action: {act}.',
    #board_game_expert have the observation xxx, try to take action:xxx and say xxx to GoodGuy, ...(echange the turn)]
    #the long history list example of opponent: ['1th Game Start', 'GoodGuy have the observation':xxx(determined by args.no_hindsight_obs), and try to take action:xxx', "GoodGuy ..."]
    #the long history list example of PEAgent: ['1th Game Start', 'board_game_expert have the observation xxx, try to take action: xxx and say xxx to GoodGuy', "board_game_expert ..."]
    
    #At the end of the game, to get summariztion and reflextion about the game    
    #if stage != "valid" and score >= 0:
    if stage != "valid":
        #rewrite long history and get reflextion of self and opponent.
        memory_summarization= ctx.robot_agents[(args.user_index + 1) % args.agents_num].get_summarization(ctx.robot_agents[(args.user_index+1) % args.agents_num].name,
            long_memory, ctx.robot_agents[args.user_index].name, no_highsight_obs = args.no_hindsight_obs)
        #In the first round of first game, name holds card1 does action .... continue ..."
        print(memory_summarization)
        
        ctx.robot_agents[(args.user_index + 1) % args.agents_num].add_long_memory(memory_summarization)
        #if args.rule_model == "suspicion-agent":
            #rule_model.add_long_memory("One of Games Started! " + memory_summarization, start_idx == args.user_index)
        
        util.get_logging(logger_name= log_file_name + '_long_memory',
                         content={str(game_idx + 1): {"long_memory": long_memory}})
        util.get_logging(logger_name= log_file_name + '_memory_summary',
                         content={str(game_idx + 1): {"memory_summary": memory_summarization}})
        
    policy = {"private":private_policy,"public":public_policy}
    
    return policy, score

def run(args):
    """
    Run IIG-Policy Evolution
    """
    settings = Settings()
    settings.model = load_model_setting(args.llm)#llm = "openai-gpt-4-0613"  #settings.model = OpenAIGPT4Settings

    # Model initialization verification
    '''
    res = util.verify_model_initialization(settings)
    if res != "OK":
        console.print(res, style="red")
        return
    '''
    
    agent_configs = []
    for idx in range(args.agents_num):
        agent_config = get_agent_config(args, idx)
        agent_configs.append(agent_config)

    # Get game rule and observation rule from game config
    try:
        game_config = util.load_json(Path(args.game_config))       
        if game_config == {}:
            console.print(
                "Empty configuration, please provide a valid one", style="red"
            )
        game_config["path"] = args.game_config
        
    except json.JSONDecodeError:
        console.print(
            "Invalid configuration, please provide a valid one", style="red"
        )
        game_config = Prompt.ask(
            "Enter the path to the game configuration file", default="./game_config.json"
        )
    #game_config = {"name": "Leduc Hold'em Poker Limit",
    #"game_rule":" the deck：two cards of King, Queen and Jack. two players, only two rounds, two-bet maximum. one public.  Raise: . Call:.
    #1 small blind, 2 big blind, one card, then betting. one public, bet again. \n Single Game Win/Draw/Lose Rule: . \n Whole Game Win/Draw/Lose Rule:. \n Winning Payoff Rule:  . \n Lose Payoff Rule:  . ",
    #"observation_rule": "The observation :dict. observation space: `'raw_legal_actions'. 'hand' .  'public_card'. game_num . all_chips .  rest_chips , opponent_rest_chips ."
    #"path":"./game_config/leduc_limit.json"}

    #Initialize PEAgent
    ctx = agi_init(agent_configs, game_config, console, settings)
    log_file_name = ctx.robot_agents[(args.user_index+1) % args.agents_num].name+'_vs_'+ctx.robot_agents[args.user_index % args.agents_num].name + '_'+args.rule_model + '_'+args.llm+'_'+args.mode
    #board_game_expert_vs_GoodGuy_cfr_openai-gpt-4-0613_automatic
    #args.user_index=1 means that the second user is opponent

    #Initialize Environment
    env = rlcard.make('leduc-holdem', config={'seed': args.seed})    
    env.reset()
    chips = [100,100]
    rule_model = get_rule_model(args.rule_model, settings.model.llm, ctx.robot_agents[args.user_index].name)
    
    if args.load_policy and os.path.exists(args.policy_record):
        with open(args.policy_record, "r") as p_f:
            p = json.load(p_f)
            best_policy = p["best_policy"]
            #print(best_policy)
    else:
        best_policy = {"private":"","public":""}

    #Game Start...
    #train and vaild for selecting the best policy    
    if args.train:         
        train_num = 100
        valid_loops = 5
        valid_batch_per_loop = 4
        valid_num = valid_loops * valid_batch_per_loop
        
        stage = "train"
        policies = {"private":[],"public":[]}
        input_train_policy = {"private":"","public":""}
        if best_policy["private"] != "":
            #policies["private"].append(best_policy["private"])
            input_train_policy["private"] = best_policy["private"]
        if best_policy["public"] != "":
            #policies["public"].append(best_policy["public"])
            input_train_policy["public"] = best_policy["public"]
            

        #with open(args.train_memory_record,"r") as tm_f:
            #ctx.robot_agents[(args.user_index+1)%args.agents_num].memory = json.load(tm_f) ["train_memories"] 
               
        #with open(args.train_policy_record,"r") as tp_f:
            #policies = json.load(tp_f) ["policies"]
            
        for game_idx in range(0, train_num):

        #for game_idx in range(train_num):
            ctx.print(f"Stage: {stage}. Game ID： {game_idx+1}.", style="red")
            
            input_train_policy, score = play_game(args, game_idx, ctx,  input_train_policy, rule_model, chips, stage, args.seed, log_file_name)
            print("=================================")
            print(score)
            print("=================================")
            policies["private"].append(input_train_policy["private"])
            policies["public"].append(input_train_policy["public"])
            
            #if (game_idx+1)%5 == 0:
            f3 = open(args.train_memory_record,"w")
            f3.write(json.dumps({"train_memories": ctx.robot_agents[(args.user_index+1) % args.agents_num].memory}, ensure_ascii =False))
            f3.close()
        
            f4 = open(args.train_policy_record,"w")
            f4.write(json.dumps({"policies": policies}, ensure_ascii =False))
            f4.close()
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Policy Evolution Agent',
        description='Playing Imperfect Information Games with LLM Based on Policy Evolution',
        epilog='Text at the bottom of help')

    parser.add_argument("--player1_config", default="./person_config/Persuader.json", help="experiments name")
    parser.add_argument("--player2_config", default="./person_config/GoodGuy.json", help="experiments name")
    parser.add_argument("--game_config", default="./game_config/leduc_limit.json",
                        help="./game_config/leduc_limit.json, ./game_config/limit_holdem.json, ./game_config/coup.json")

    parser.add_argument("--seed", type=int, default=1, help="random_seed")

    parser.add_argument("--llm", default="DeepSeek", help="environment flag, openai-gpt-4-0613 or openai-gpt-3.5-turbo or Qwen or Llama")
    parser.add_argument("--rule_model", default="nfsp", help="rule model: cfr or nfsp or dqn or dmc or suspicion-agent")
    parser.add_argument("--mode", default="second_tom", help="inference mode: normal or first_tom or second_tom or automatic")
    #parser.add_argument("--stage", default="train", help="train or valid or test stage")

    #user stands for opponent
    parser.add_argument("--agents_num", type=int, default=2)
    parser.add_argument("--user", default=True, help="one of the agents is baseline mode, e.g. cfr, nfsp")
    parser.add_argument("--verbose_print", action="store_true",help="""The retriever to fetch related memories.""")
    parser.add_argument("--user_index", type=int, default=1, help="user position: 0 or 1")
    parser.add_argument("--game_num", type=int, default=100)#train:40 valid:20 test:100 每4条一训练，2条进行验证(分先后手)
    parser.add_argument("--random_seed", default=True)
    parser.add_argument("--no_hindsight_obs", default=False, help = "indicates whether to add the opponent’s observation into the single game history after the end of each game")

    parser.add_argument("--train", default = True)
    parser.add_argument("--load_policy", default = False)
    parser.add_argument("--load_memory", default = False)
    parser.add_argument("--policy_record", default="./leduc_dmc_ds_policy_secondtom.json", help="experiments name")
    parser.add_argument("--memory_record", default="./leduc_dmc_ds_memory_secondtom.json", help="experiments name")
    parser.add_argument("--train_memory_record", default="./leduc_ds_qwen_trainmemory_secondtom.json", help="experiments name")
    parser.add_argument("--train_policy_record", default="./leduc_ds_qwen_trainpolicy_secondtom.json", help="experiments name")

    args = parser.parse_args()
    run(args)