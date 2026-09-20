import time
# import timeout_decorator  # Commented out for Windows compatibility
'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file. If you do so, measure the
    running time of __init__, new_game, update_game, and get_actions yourself
    (for example, with time.perf_counter). This local workaround does not relax
    the one-second limit: it is a hard constraint and will be enforced
    independently during marking.
'''
from agent_baselines import Agent

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from agent_baselines.py first.
    
    You can add/override attributes and methods as needed.
    '''

    # @timeout_decorator.timeout(1)  # Commented out for Windows compatibility
    def __init__(self, agent_name='Give a nickname'):
        super().__init__(agent_name)

        '''Implement your agent here.'''

    # @timeout_decorator.timeout(1)  # Commented out for Windows compatibility
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        '''Implement your agent here.'''

    # @timeout_decorator.timeout(1)  # Commented out for Windows compatibility
    def update_game(self, all_power_orders):
        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    # @timeout_decorator.timeout(1)  # Commented out for Windows compatibility
    def get_actions(self):

        '''Implement your agent here.'''
        
        return [] 