# PolicyEvol
PolicyEvol for card games by LLM-based agents

## Game Description
### 1. Leduc Hold’em
Leduc Hold’em is first introduced in Southey et al. (2005) and sometimes used in academic research. It is played with a deck consisting only two cards of King, Queen and Jack, six cards in total. Each game is fixed with two players, only two rounds, and two-bet maximum in each round. The game begins with each player being dealt one card privately, followed by a betting round. Then, another card is dealt faceup as a community (or board) card, and there is another betting round. Finally, the players reveal their private cards. If one player’s private card is the same rank as the board card, he or she wins the game; otherwise, the player whose private card has the higher rank wins.
### 2. BlackJack
Blackjack, also known as 21, is a popular card game that involves a dealer and a player. Players must decide whether to hit or stand based on their own hand, the dealer’s face-up card, and the dealer’s one hidden card. The objective is to beat the dealer without exceeding 21 points. For this game, we observe whether LLM-based agents can make rational decisions under uncertain scenarios.
## Get API KEY
[DeepSeek API Documentation](https://api-docs.deepseek.com/zh-cn/) 

[Qwen API Documentation](https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api)
## Running
### 1. Leduc Hold’em 
To start Leduc Hold’em games competing, run code example as:

``python main.py --rule_model [cfr/ nfsp / dqn / dmc / suspicion-agent] --llm [Qwen / DeepSeek]``
### 2. BlackJack
To start BlackJack games competing, please utilize the following code snippet:

```
from play_blackjack_game import play

if __name__ == "__main__":

    number_of_game = 2
  
    model = 'DeepSeek' 
  
    game_style = 'evolution'
  
    storage_name = "Deepseek play Blackjack with evolution"
  
    play(number_of_game,model,game_style,storage_name)
```

Part of code is borrowed from [Suspicion-Agent](https://github.com/CR-Gjx/Suspicion-Agent) and Agent-Pro(https://github.com/zwq2018/Agent-Pro)
## Result
Methodology and Experiments can be refered to our paper [PolicyEvolAgent](https://arxiv.org/pdf/2504.15313)
