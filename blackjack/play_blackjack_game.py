import json
import random
import rlcard
import re
from rlcard.agents import RandomAgent
from LLMAPIs import GPT35API
from LLMAPIs import GPT4API
from LLMAPIs import llama2_70b_chatAPI
from LLMAPIs import QwenAPI
from LLMAPIs import DeepSeekAPI
import rlcard.envs


class Round:
    round = 0
    num_hit = 0
    num_stand = 0

    @staticmethod
    def reset():
        Round.round = 0


def play(game_num, model, game_style, storage_name):
    Round.reset()
    experience = []
    record = []

    class LlmAgent(RandomAgent):

        def __init__(self, num_actions):
            super().__init__(num_actions)

        @staticmethod
        def step(state):
            Round.round += 1
            deal_card = state['raw_obs']['dealer hand']
            hand_card = state['raw_obs']['player0 hand']
            llm = find_model(model)
            p = []
            begin_info = "You are a player in blackjack. Please beat the dealer and win the game.\n"
            game_rule = "Game Rule:\n1. Please try to get your card total to as close to 21 as possible, without going over, and still having a higher total than the dealer.\n2. If anyone's point total exceeds 21, he or she loses the game. \n3. You can only choose one of the following two actions: {\"Stand\", \"Hit\"}. If you choose to Stand, you will stop taking cards and wait for the dealer to finish. If you choose to Hit, you can continue to take a card, but there is also the risk of losing the game over 21 points. \n4. After all players have completed their hands, the dealer reveals their hidden card. Dealers must hit until their cards total 17 or higher.\n"
            game_info = "The dealer's current card is {" + card2string(
                deal_card
            ) + "}. The dealer has another hidden card. You don't know what it is. Your current cards are {" + card2string(
                hand_card) + "}. "

            if game_style == 'Vanilla':
                p.append({"role": "system", "content": begin_info + game_rule})
                game_info += "Please output your action in following format: ###My action is {your action}, without any other text."
                p.append({"role": "user", "content": game_info})
            if game_style == 'Radical':
                begin_info = "You are an aggressive player of blackjack who likes to take risks to earn high returns. Please beat the dealer and win the game."
                p.append({"role": "system", "content": begin_info + game_rule})
                game_info += "Please output your action in following format: ###My action is {your action}, without any other text."
                p.append({"role": "user", "content": game_info})
            if game_style == 'ReAct':
                p.append({"role": "system", "content": begin_info + game_rule})
                game_info += "Please first think and reason about the current hand and then generate your action as follows: ###My thought is {Your Thought}. My action is {your action}."
                p.append({"role": "user", "content": game_info})
            if game_style == 'ReFlexion':
                p.append({"role": "system", "content": begin_info + game_rule})
                game_info += "Please first think and reason about the current hand and then generate your action as follows: ###My thought is {Your Thought}. My action is {your action}."
                p.append({"role": "user", "content": game_info})
                llm_res = llm.response(p)
                p.append({"role": "assistant", "content": llm_res})
                reflexion_info = "Please carefully check the response you just output, and then refine your answer . The final output is also in following format: ###My thought is {Your Thought}. My action is {your action}."
                p.append({"role": "user", "content": reflexion_info})
            if game_style == 'agentpro':
                begin_info = "You are an aggressive player of blackjack who likes to take risks to earn high returns. Please beat the dealer and win the game."
                p.append({"role": "system", "content": begin_info + game_rule})
                game_info += "Please read the behavoiral guideline and world modeling carefully . Then you should analyze your own cards and your strategies in Self-belief and then analyze the dealer cards in World-belief. Lastly, please select your action from {\"Stand\",\"Hit\"}.### Output Format: Self-Belief is {Belief about youself}. World-Belief is {Belief about the dealer}. My action is {Your action}. Please output in the given format."
                p.append({"role": "user", "content": game_info})
                
            if game_style == 'evolution':
                p.append({"role": "system", "content": begin_info + game_rule})
                game_info += "Please first think and reason about the your own cards and your strategies (like aggresive or conservative), and also analyze the dealer cards, then generate your action as follows: ###My thought is {Your Thought}. My action is {your action}.\n"
                if "".join(experience[-2:]) != "":
                    game_info += "Before you make your action, you can refer to the suggestions below to earn high returns if needed: "+" ".join(experience[-2:])
                p.append({"role": "user", "content": game_info})
                #llm_res_ = llm.response(p)
                #p.append({"role": "assistant", "content": llm_res_})
                #reflexion_info = "Please carefully check the response you just output, and then refine your answer, and finally give a suggestion or guideline in one or two sentences based on this game experience for better winning dealer in the next few games. The final output is also in following format: ###My thought is {Your Thought}. My action is {your action}. My suggestion is <SUGGESTION>{your suggestion}</SUGGESTION>"
                #reflexion_info = "Please carefully check the response you just output, and then give a suggestion or guideline in one or two sentences based on this game experience for better winning dealer in the next few games. The final output is also in following format: ###My thought is {Your Thought}. My suggestion is <SUGGESTION>{your suggestion}</SUGGESTION>"
                #p.append({"role": "user", "content": reflexion_info})
                              
            llm_res = llm.response(p)
            p.append({"role": "assistant", "content": llm_res})
            filename = storage_name + '.json'
            with open(filename, 'a') as file:
                json.dump(p, file, indent=4)
                # file.write(json_str + '\n')
            if game_style == 'evolution':
                #print(llm_res)
                #experience.append(extract_experience(llm_res))
                #print(experience)
                record = p
                choice = -1
                if extract_choice(llm_res) == "hit":
                    choice = 0
                elif extract_choice(llm_res) == "stand":
                    choice = 1
                else:
                    choice = -1
                return choice
                           
            choice = -1
            if extract_choice(llm_res) == "hit":
                choice = 0
            elif extract_choice(llm_res) == "stand":
                choice = 1
            else:
                choice = -1
            return choice

    def find_model(model):
        if model == "gpt-3.5":
            return GPT35API()
        if model == "gpt-4":
            return GPT4API()
        if model == "Llama70b":
            return llama2_70b_chatAPI()
        if model == "Qwen":
            return QwenAPI()
        if model == "DeepSeek":
            return DeepSeekAPI()
        
    def extract_experience(text):
        #text = to_lower(text)
        pattern = r'<SUGGESTION>(.*?)</SUGGESTION>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        else:
            return ''

    def extract_choice(text):
        text = to_lower(text)
        last_hit_index = text.rfind("hit")
        last_stand_index = text.rfind("stand")
        if last_hit_index > last_stand_index:
            return "hit"
        elif last_stand_index > last_hit_index:
            return "stand"
        else:
            return None

    def to_lower(str):
        lowercase_string = str.lower()
        return lowercase_string

    def card2string(cardList):
        str = ''
        str = ','.join(cardList)
        str = str.replace('C', 'Club ')
        str = str.replace('S', 'Spade ')
        str = str.replace('H', 'Heart ')
        str = str.replace('D', 'Diamond ')
        str = str.replace('T', '10')
        return str

    num_players = 1
    env = rlcard.make('blackjack',
                      config={
                          'game_num_players': num_players,
                          "seed": random.randint(0, 10**10)
                      })

    llm_agent = LlmAgent(num_actions=env.num_actions)

    env.set_agents([llm_agent])

    def play_game(env, experience):
        trajectories, payoffs = env.run(is_training=False)
        if len(trajectories[0]) != 0:
            final_state = []
            action_record = []
            state = []

            for i in range(num_players):
                final_state.append(trajectories[i][-1])
                state.append(final_state[i]['raw_obs'])

                action_record.append(final_state[i]['action_record'])
                print(final_state[i]['action_record'])
                if game_style == 'evolution':
                    actions = ','.join([a[1] for a in final_state[i]['action_record']])
                    reflection_prompt = f"The action state of yours is : {actions}."
                    
        Round.round += 1
        res_str = ('dealer {}, '.format(state[0]['state'][1]) +
                   'player {}, '.format(state[0]['state'][0]))
        if payoffs[0] == 1:
            final_res = "win."
        elif payoffs[0] == 0:
            final_res = "draw."
        elif payoffs[0] == -1:
            final_res = "lose."
        p = ({"final cards": res_str, "final results": final_res})
        print(p)
        if game_style == 'evolution':
            reflection_prompt += f"The final cards of dealer and you are {res_str}, and final results is: you {final_res}."
            reflection_prompt += "Please carefully check the game process, and then give a suggestion or guideline in one or two sentences based on this game experience and result for future to better win dealer. The final output is also in following format: ###My thought is {Your Thought}. My suggestion is <SUGGESTION>{your suggestion}</SUGGESTION>"
            record.append({"role": "user", "content": reflection_prompt})
            llm = find_model(model)
            llm_res_ = llm.response(record)
            experience.append(extract_experience(llm_res_))
        with open(storage_name + ".json", 'a', encoding='utf-8') as file:
            json.dump(p, file, indent=4)
        return experience[-2:]

    num_matches = game_num
    #experience = ["to be more cautious when deciding to hit, especially if the dealer's visible card is high, as hitting can easily lead to going over 21 and losing", "to carefully consider your decision to stand or hit based on the dealer's visible card and your own hand, especially when both totals are close, to avoid draws and potentially win more often."]
    

    for i in range( num_matches):
        record = []
        print("=======================================")
        print(i)
        print(experience)
        experience = play_game(env, experience)
        print("=======================================")
        
if __name__ == "__main__":
    global experience, record 
    play(100, 'DeepSeek', 'ReFlexion', "DeepSeek-ReFlexion-BlackJack")
