"""
Group 2 - experiments supporting the report.

    BasicAStarAgent  -> Basic technique (per-unit A*, nearest-SC targeting)
    Technique1Agent  -> + context-aware target selection
    Technique2Agent  -> + joint-order planning (replaces per-unit A*)
    StudentAgent     -> + opponent modelling and best response (final agent)

Experiments
    ablation   : every configuration in Scenario 1 and Scenario 2
    scenario3  : Scenario 3 stand-in. Opponents are drawn from the Scenario 2
                 pool, but exactly one opponent per game is a strong agent
                 standing in for the unavailable Hidden Agent (default: our
                 Technique 1 agent). Also runs the final agent with the
                 best-response step disabled, to separate the two parts of
                 Technique 3.
    scenario4  : Scenario 4 stand-in. A multi-round tournament in which our four
                 configurations and three baselines play in the same games, with
                 power assignments rotated so every contender plays every power
                 equally often. Reports mean SCs, mean rank and top-3 rate.
    selfplay   : stress test with seven copies of the final agent in one game
                 (stability and timing when every opponent is a search agent).
    Worst-case get_actions() time is recorded for every configuration.

Usage
    python test_2.py                                # everything, default sizes
    python test_2.py --exp ablation --repeats 20
    python test_2.py --exp scenario3 --proxy t2     # different stand-in
    python test_2.py --exp scenario4 --rounds 10
"""
import argparse
import random
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from game import run_one_game
from agent_baselines import StaticAgent, RandomAgent, GreedyAgent, AttitudeAgent
from agent_2 import StudentAgent, BasicAStarAgent, Technique1Agent, Technique2Agent


class Technique3NoBestResponse(StudentAgent):
    """Final agent with the enemy best-response step switched off (test-only
    variant). Keeps the scenario mixture and history observation."""
    def __init__(self, agent_name="Technique3NoBestResponse"):
        super().__init__(agent_name)
        self.BR_WEIGHT = 0.0

ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

CONFIGS = [
    ('Basic', BasicAStarAgent),
    ('T1 targeting', Technique1Agent),
    ('T2 joint planning', Technique2Agent),
    ('T3 final', StudentAgent),
]

SCENARIO_1_POOL = [StaticAgent]
# Same pool as the provided test.py: Random is less likely than the other two.
SCENARIO_2_POOL = [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent]


###############################################################################
# Timing: wrap an agent class so every get_actions() call is timed.
###############################################################################
TIMINGS = defaultdict(list)


def timed(cls, label):
    class Timed(cls):
        def get_actions(self):
            start = time.perf_counter()
            orders = super().get_actions()
            TIMINGS[label].append(time.perf_counter() - start)
            return orders
    Timed.__name__ = cls.__name__
    return Timed


###############################################################################
# Scoring and experiment loop (adapted from the provided test.py).
###############################################################################
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


def experiment(player_agent, opponent_agent_pool, repeat_nums=10, strong_opponent=None,
               desc=''):
    """Play repeat_nums games as each power. If strong_opponent is given, exactly
    one opponent per game is that agent (Scenario 3 stand-in)."""
    all_scores = defaultdict(list)
    all_wins = defaultdict(list)

    with tqdm(total=repeat_nums * len(ALL_POWERS), desc=desc) as pbar:
        for _ in range(repeat_nums):
            for me in ALL_POWERS:
                others = [p for p in ALL_POWERS if p != me]
                strong_power = random.choice(others) if strong_opponent else None
                agents_dict = {}
                for p in ALL_POWERS:
                    if p == me:
                        agents_dict[p] = player_agent()
                    elif p == strong_power:
                        agents_dict[p] = strong_opponent()
                    else:
                        agents_dict[p] = random.choice(opponent_agent_pool)()
                results, _ = run_one_game(agents_dict)
                scores, wins = scoring(results)
                for key in (me, 'ALL'):
                    all_scores[key].append(scores[me])
                    all_wins[key].append(wins[me])
                pbar.update(1)

    summary = {}
    for k in all_scores:
        n = len(all_wins[k])
        summary[k] = {
            'sc_mean': float(np.mean(all_scores[k])),
            'sc_std': float(np.std(all_scores[k])),
            'win': 100.0 * sum(w == 'WIN' for w in all_wins[k]) / n,
            'survive': 100.0 * sum(w == 'SURVIVE' for w in all_wins[k]) / n,
            'defeat': 100.0 * sum(w == 'DEFEAT' for w in all_wins[k]) / n,
        }
    return summary


def print_per_power(title, summary):
    print(f'----- {title} -----')
    for p in ALL_POWERS + ['ALL']:
        s = summary[p]
        print(f"{p}: SCs - {s['sc_mean']:.2f}±{s['sc_std']:.2f}, Wins - {s['win']:.2f}%, "
              f"Survives - {s['survive']:.2f}%, Defeats - {s['defeat']:.2f}%")


def print_table(title, rows):
    """rows: list of (label, summary_dict) using the 'ALL' entry."""
    print(f'\n===== {title} =====')
    print(f"{'Configuration':<20}{'SCs':>14}{'Win %':>9}{'Survive %':>11}{'Defeat %':>10}")
    for label, summary in rows:
        s = summary['ALL']
        sc = f"{s['sc_mean']:.2f}±{s['sc_std']:.2f}"
        print(f"{label:<20}{sc:>14}{s['win']:>9.2f}{s['survive']:>11.2f}{s['defeat']:>10.2f}")


def print_timings():
    print('\n===== get_actions() time per call (seconds) =====')
    for label, times in TIMINGS.items():
        if times:
            print(f'{label:<20} mean {np.mean(times):.3f}   '
                  f'p99 {np.percentile(times, 99):.3f}   max {max(times):.3f}   '
                  f'calls {len(times)}')


###############################################################################
# Experiments
###############################################################################
def run_ablation(repeats_s1, repeats_s2):
    """Basic -> T1 -> T2 -> T3 in Scenarios 1 and 2.

    Note: Scenario 1 opponents never move, and the A* agents are deterministic,
    so Scenario 1 repeats of the A* configurations replay the same game."""
    s1_rows, s2_rows = [], []
    for label, cls in CONFIGS:
        agent = timed(cls, label)
        s1 = experiment(agent, SCENARIO_1_POOL, repeats_s1, desc=f'S1 {label}')
        print_per_power(f'Scenario 1 - {label}', s1)
        s1_rows.append((label, s1))
        s2 = experiment(agent, SCENARIO_2_POOL, repeats_s2, desc=f'S2 {label}')
        print_per_power(f'Scenario 2 - {label}', s2)
        s2_rows.append((label, s2))
    print_table('Scenario 1 (Static opponents)', s1_rows)
    print_table('Scenario 2 (Random / Attitude / Greedy opponents)', s2_rows)


PROXIES = {'t1': ('T1 targeting', Technique1Agent),
           't2': ('T2 joint planning', Technique2Agent),
           'final': ('T3 final', StudentAgent)}


def run_scenario3(repeats, proxy):
    """Scenario 3 stand-in. Exactly one opponent per game is the strong proxy,
    the other six come from the Scenario 2 pool.

    The real Hidden Agent wins ~50% of games in Scenario 2. Compare the proxy's
    own Scenario 2 win rate (ablation experiment) with that figure to judge how
    much weaker or stronger it is than the Hidden Agent."""
    proxy_label, proxy_cls = PROXIES[proxy]
    strong = timed(proxy_cls, 'proxy: ' + proxy_label)
    contenders = CONFIGS[:3] + [('T3 without best response', Technique3NoBestResponse),
                                CONFIGS[3]]
    rows = []
    for label, cls in contenders:
        agent = timed(cls, label)
        res = experiment(agent, SCENARIO_2_POOL, repeats, strong_opponent=strong,
                         desc=f'S3-proxy {label}')
        print_per_power(f'Scenario 3 stand-in ({proxy_label} as strong opponent) - {label}', res)
        rows.append((label, res))
    print_table(f'Scenario 3 stand-in (one {proxy_label} opponent per game)', rows)


###############################################################################
# Scenario 4 stand-in: multi-round tournament.
###############################################################################
def average_ranks(scores):
    """Rank 1 = most supply centres. Ties share the average of the ranks they span."""
    ranks = {}
    for k, v in scores.items():
        better = sum(1 for x in scores.values() if x > v)
        equal = sum(1 for x in scores.values() if x == v)
        ranks[k] = better + (equal + 1) / 2.0
    return ranks


def run_scenario4(rounds):
    """Each round shuffles the seven contenders and plays 7 games, rotating the
    assignment of contenders to powers so every contender plays every power once.

    A contender is top-3 when its (tie-averaged) rank is <= 3, so a three-way tie
    for third does not count as top-3.

    Limitation: the real Scenario 4 opponents are other groups' agents, which are
    likely stronger than the three baselines used here."""
    ours = [c for c in CONFIGS]
    contenders = ours + [('Greedy', GreedyAgent), ('Attitude', AttitudeAgent),
                         ('Random', RandomAgent)]
    labels = [c[0] for c in contenders]
    factories = {label: (timed(cls, label) if (label, cls) in ours else cls)
                 for label, cls in contenders}
    our_labels = [c[0] for c in ours]

    sc = defaultdict(list)
    rank = defaultdict(list)
    rank_ours = defaultdict(list)
    wins = defaultdict(int)
    top3 = defaultdict(int)
    games = 0
    failed = 0

    with tqdm(total=rounds * len(ALL_POWERS), desc='S4 tournament') as pbar:
        for _ in range(rounds):
            order = labels[:]
            random.shuffle(order)
            for shift in range(len(ALL_POWERS)):
                agents_dict, power_of = {}, {}
                for i, label in enumerate(order):
                    power = ALL_POWERS[(i + shift) % len(ALL_POWERS)]
                    agents_dict[power] = factories[label]()
                    power_of[label] = power
                try:
                    results, _ = run_one_game(agents_dict)
                except Exception as exc:          # count, do not hide
                    failed += 1
                    print(f'\n[scenario4] game failed: {type(exc).__name__}: {exc}')
                    pbar.update(1)
                    continue
                games += 1
                scores = {label: results[power_of[label]] for label in labels}
                r = average_ranks(scores)
                r_ours = average_ranks({l: scores[l] for l in our_labels})
                for label in labels:
                    sc[label].append(scores[label])
                    rank[label].append(r[label])
                    wins[label] += scores[label] >= 18
                    top3[label] += r[label] <= 3
                for label in our_labels:
                    rank_ours[label].append(r_ours[label])
                pbar.update(1)

    print(f'\n===== Scenario 4 stand-in: {games} games, {failed} failed =====')
    if not games:
        return
    print(f"{'Contender':<20}{'SCs':>14}{'Mean rank':>11}{'Top-3 %':>9}{'Win %':>8}"
          f"{'Rank (ours)':>13}")
    for label in labels:
        scs = f"{np.mean(sc[label]):.2f}±{np.std(sc[label]):.2f}"
        own = f"{np.mean(rank_ours[label]):.2f}" if label in our_labels else '-'
        print(f"{label:<20}{scs:>14}{np.mean(rank[label]):>11.2f}"
              f"{100.0 * top3[label] / games:>9.1f}{100.0 * wins[label] / games:>8.1f}"
              f"{own:>13}")
    print('Rank (ours) = mean rank considering only our four configurations (1 = best of 4).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['all', 'ablation', 'scenario3', 'scenario4',
                                          'selfplay'], default='all')
    parser.add_argument('--repeats', type=int, default=10,
                        help='games per power in Scenario 2 and the Scenario 3 stand-in')
    parser.add_argument('--repeats-s1', type=int, default=10,
                        help='games per power in Scenario 1 (deterministic opponents)')
    parser.add_argument('--proxy', choices=list(PROXIES), default='t1',
                        help='strong opponent used as the Hidden Agent stand-in')
    parser.add_argument('--rounds', type=int, default=10,
                        help='tournament rounds in the Scenario 4 stand-in (7 games each)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.exp in ('all', 'ablation'):
        run_ablation(args.repeats_s1, args.repeats)
    if args.exp in ('all', 'scenario3'):
        run_scenario3(args.repeats, args.proxy)
    if args.exp in ('all', 'scenario4'):
        run_scenario4(args.rounds)
    print_timings()