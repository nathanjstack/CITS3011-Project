import random
import numpy as np
from game import run_one_game
from agent_baselines import StaticAgent, RandomAgent, GreedyAgent, AttitudeAgent
from agent_groupnumber import StudentAgent

# This file provides an example to simulate one game and export the game process for visualization.

if __name__ == "__main__":

	agents_dict = {
		'AUSTRIA': StaticAgent(), 
		'ENGLAND': StaticAgent(), 
		'FRANCE': StaticAgent(), 
		'GERMANY': StaticAgent(), 
		'ITALY': StaticAgent(), 
		'RUSSIA': StaticAgent(), 
		'TURKEY': GreedyAgent()
	}

	run_one_game(agents_dict, save_file='game_for_vis.json')

	'''
	A JSON file will be saved, which can be visualized using the Web Interface provided on https://github.com/diplomacy/diplomacy?tab=readme-ov-file#web-interface
	
	Follow the instructions to setup the Web Interface: https://github.com/diplomacy/diplomacy?tab=readme-ov-file#web-interface, which may take some time. 

	Then click 'load a game from the disk' on the Web Interface, to load and visualize the saved JSON.

	This visualization is not necessary for completing the project. It is mainly to assist the debugging and for fun.

	Note that if there is an existing file with the same name, the game data will be appended to the same file, and causing errors for visualization.
	'''
 
