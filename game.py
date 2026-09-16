import diplomacy
from diplomacy import Game

def copy_game(game):
    '''This is a deepcopy, you should always use this to make a copy of the game engine object'''
    dic = diplomacy.utils.export.to_saved_game_format(game)
    game_copy = diplomacy.utils.export.from_saved_game_format(dic)
    return game_copy

def run_one_game(agents_dict, game=None, end_year=1920, save_file=None):
    '''
    This is to simulation one game. Do not change anything.

    Inputs:
        agents_dict: A dictionary, with power names as keys, and Agent objects as values.
        game:        A Game object as the initial state. If none, a new game is created. The default standard map is used.
        end_year:    The maximum end year of the game.
        save_file:   Export the game process to a file for visualize or analysis (the output is appended to the file if it is an existing file).

    Outputs:
        results:     A dictionary, with power names as keys, and the number of supply centers as values.
        year:        The actual end year.

    '''

    if game == None:
        game = Game() 
    
    for p in agents_dict.keys():
        game_copy = copy_game(game) # Copies of the game are passed to agents, so making change to the game within the Agent does not affect outside.
        agents_dict[p].new_game(game_copy, power_name=p)
    
    year = None

    while not game.is_game_done:

        # The game will terminate in end_year, if no winner before that
        current_phase = game.get_current_phase()
        year = current_phase[1:5]
        if int(year) >= end_year:
            break       

        # Get actions from agents
        all_power_orders = {}
        for p in game.powers.keys():
            all_power_orders[p] = agents_dict[p].get_actions()

        # Processing the game to move to the next phase
        for p in game.powers.keys():
            game.set_orders(p, all_power_orders[p])
        game.process()

        # Update the internal states of the agents
        for p in agents_dict.keys():
            agents_dict[p].update_game(all_power_orders)

    if save_file:
        diplomacy.utils.export.to_saved_game_format(game, output_path=save_file)

    results = {p: len(game.powers[p].centers) for p in game.powers.keys()} # >= 18 supply centers to win

    if not year:
        year = end_year

    return results, year

