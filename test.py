import random
import numpy as np
from tqdm import tqdm
from game import run_one_game
from collections import defaultdict
from agent_baselines import StaticAgent, RandomAgent, GreedyAgent, AttitudeAgent
from agent_groupnumber import StudentAgent

# This file provides examples for you to test the performance of your agents. The testing code may be different during the marking.

ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

def scoring(centres):
    scores = {k: min(v, 18) for k, v in centres.items()}
    wins = {}
    for k, v in scores.items():
        if v == 18:
            wins[k] = 'WIN'
        elif v == 0:
            wins[k] = 'DEFEAT'
        else:
            wins[k] = 'SURVIVE'
    return scores, wins

def experiment(player_agent, opponent_agent_pool, repeat_nums=10):
    all_scores = defaultdict(list)
    all_wins = defaultdict(list)
    total_games = repeat_nums * len(ALL_POWERS)

    with tqdm(total=total_games) as pbar:
        for r in range(repeat_nums):
            for i in ALL_POWERS:
                agents_dict = {}
                for p in ALL_POWERS:
                    if p == i:
                        agents_dict[p] = player_agent()
                    else:
                        opponent_agent = random.choice(opponent_agent_pool)
                        agents_dict[p] = opponent_agent()
                results, _ = run_one_game(agents_dict)
                scores, wins = scoring(results)
                all_scores[i].append(scores[i])
                all_scores['ALL'].append(scores[i])
                all_wins[i].append(wins[i])
                all_wins['ALL'].append(wins[i])
                pbar.update(1)

    scores_avg = {k: np.mean(v) for k, v in all_scores.items()}
    scores_std = {k: np.std(v) for k, v in all_scores.items()}

    win_rates, survive_rates, defeat_rates = {}, {}, {}
    for k, v in all_wins.items():
        win_rates[k] = sum([i == 'WIN' for i in v]) / len(v)
        survive_rates[k] = sum([i == 'SURVIVE' for i in v]) / len(v)
        defeat_rates[k] = sum([i == 'DEFEAT' for i in v]) / len(v)
        win_rates[k] = round(win_rates[k] * 100, 2)
        survive_rates[k] = round(survive_rates[k] * 100, 2)
        defeat_rates[k] = round(defeat_rates[k] * 100, 2)

    print('----- The Per-Power and the Overall Performance of the Player Agent -----')
    for p in ALL_POWERS + ['ALL']:
        print(f'{p}: SCs - {round(scores_avg[p], 2)}±{round(scores_std[p], 2)}, Wins - {win_rates[p]}%, Survives - {survive_rates[p]}%, Defeats - {defeat_rates[p]}%')
    print('-------------------------------------------------------------------------')


if __name__ == "__main__":

    print('Evaluating Scenario 1 ...')
    experiment(player_agent=StudentAgent, opponent_agent_pool=[StaticAgent], repeat_nums=10)

    print('Evaluating Scenario 2 ...')
    experiment(player_agent=StudentAgent, opponent_agent_pool=[RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent], repeat_nums=10)
