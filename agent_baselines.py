import random
import networkx as nx
from game import run_one_game

'''This file provides the abstract Agent class, and baseline agents. Do not change anything.'''

class Agent:
    '''An abstract class for all agents. Your agent should inherit this class.'''

    def __init__(self, agent_name):
        '''
        Initialises the agent, and gives it a name.
        '''
        self.agent_name = agent_name

    def new_game(self, game, power_name):
        '''
        Initialises a new game. You can maintain and reset more state variables here.
        '''

        # the current game object (a diplomacy.engine.game object). Read the doc for more details, you can extract all the needed information from this object.
        self.game = game 

        # the power currently under control
        self.power_name = power_name 

    def update_game(self, all_power_orders):
        '''
        Update the internal game state. You can maintain and update more state variables here.
        '''

        # the internal game object is a different instance from the game object in the simulation engine. 
        # the following codes are to keep them consistent. do not make changes.
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process() 

    def get_actions(self):
        '''
        Return actions. The output is a list of orders.
        '''
        return []


######################### Baseline Agents ##############################

class StaticAgent(Agent):
    '''An agent is static and can only Hold.'''

    def __init__(self, agent_name='Static Agent'):
        super().__init__(agent_name)

    def get_actions(self):
        return []


class RandomAgent(Agent):
    '''An agent is fully random.'''

    def __init__(self, agent_name='Random Agent'):
        super().__init__(agent_name)

    def get_actions(self):

        # Getting the list of possible orders for all locations
        possible_orders = self.game.get_all_possible_orders()
        
        # Randomly sampling a valid order
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = [random.choice(possible_orders[loc]) for loc in orderable_locations if possible_orders[loc]]
        
        return power_orders 


class AttitudeAgent(Agent):
    '''An agent can be hostile, neutral, or friendly'''

    def __init__(self, agent_name='Attitude Agent'):
        super().__init__(agent_name)
        self.transition_probs = { # the attitude depends on the last turn actions and the previous attitude, based on the following transition probabilities. the check order matters.
            'AttackedBySuccess':  {'FRIENDLY': [0.0, 0.2, 0.8], 'NEUTRAL': [0.0, 0.1, 0.9], 'HOSTILE': [0.0, 0.0, 1.0]},
            'AttackedBy':         {'FRIENDLY': [0.0, 0.5, 0.5], 'NEUTRAL': [0.0, 0.3, 0.7], 'HOSTILE': [0.0, 0.0, 1.0]},
            'SupportedBySuccess': {'FRIENDLY': [1.0, 0.0, 0.0], 'NEUTRAL': [0.6, 0.4, 0.0], 'HOSTILE': [0.1, 0.7, 0.2]},
            'SupportedBy':        {'FRIENDLY': [1.0, 0.0, 0.0], 'NEUTRAL': [0.2, 0.7, 0.1], 'HOSTILE': [0.0, 0.3, 0.7]},
            'Other':              None, # other situations will not change the attitude
        }

    def new_game(self, game, power_name):
        '''initial attitude will be friendly'''
        self.game = game
        self.power_name = power_name
        self.attitude = {}
        for p in self.game.powers.keys():
            if p != self.power_name:
                self.attitude[p] = 'FRIENDLY'
        self.attitude[self.power_name] = 'NEUTRAL' # always be neutral to self

    def update_attitude(self, all_power_orders, self_locs, self_units):
        for p in self.game.powers.keys():
            if p == self.power_name:
                continue

            flags = {
                'AttackedBySuccess':False,
                'AttackedBy':False,
                'SupportedBySuccess':False,
                'SupportedBy':False,
                'Other':True
            }

            power_orders = all_power_orders[p]
            order_status = self.game.get_order_status(power_name=p)

            for order in power_orders:
                if '-' not in order and ' S ' not in order: # only consider move/attack/support orders
                    continue
                words = order.split(' ')
                unit = ' '.join(words[:2])

                # check Attacks
                if '-' in order and (len(words) == 4 or len(words) == 5): # 5 for via convoy
                    att_target = words[words.index('-')+1]
                    if att_target in self_locs:
                        flags['AttackedBy'] = True
                        if order_status[unit] == []: # the order was successful
                            flags['AttackedBySuccess'] = True
                        
                # check Supports
                if ' S ' in order:
                    supported_unit = ' '.join(words[words.index('S')+1:words.index('S')+3])
                    if supported_unit in self_units:
                        flags['SupportedBy'] = True
                        if order_status[unit] == []: # the order was successful
                            flags['SupportedBySuccess'] = True

            # update attitudes, the order of checks matters
            for i in ['AttackedBySuccess', 'AttackedBy', 'SupportedBySuccess', 'SupportedBy']:
                if flags[i]:
                    probs = self.transition_probs[i][self.attitude[p]]
                    self.attitude[p] = random.choices(['FRIENDLY', 'NEUTRAL', 'HOSTILE'], weights=probs, k=1)
                    self.attitude[p] = self.attitude[p][0]
                    break

        self.attitude[self.power_name] = 'NEUTRAL'

    def update_game(self, all_power_orders):
        if self.game.phase_type != 'M': # only update attitue if this is a movement phase
            update_att_flag = False
        else:
            update_att_flag = True
            self_centers_locs = self.game.get_centers(self.power_name)          # center locations before movement
            self_unit_locs = self.game.get_orderable_locations(self.power_name) # unit locations before movement
            self_locs = self_centers_locs + self_unit_locs
            self_locs = list(set(self_locs))                                    # only attacks and supports on these locations will be considered
            self_units = self.game.get_units(power_name=self.power_name)

        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

        if update_att_flag:
            self.update_attitude(all_power_orders, self_locs, self_units)

    def get_actions(self):

        all_possible_orders = self.game.get_all_possible_orders()
        all_orderable_locations = {p: self.game.get_orderable_locations(p) for p in self.game.powers.keys()}

        # get all units and locations of all powers
        locs_of_powers = {}
        units_of_powers = {}
        for p in self.game.powers.keys():
            centers_locs = self.game.get_centers(p)
            unit_locs = all_orderable_locations[p]
            locs = centers_locs + unit_locs
            locs = list(set(locs))
            for i in locs:
                locs_of_powers[i] = p
            for i in self.game.get_units(p):
                units_of_powers[i] = p

        power_orders = []
        for loc in all_orderable_locations[self.power_name]:
            possible_orders = all_possible_orders[loc]
            if possible_orders:
                filtered_orders = [] # filter orders based on attitudes
                for order in possible_orders:
                    words = order.split(' ')
                    unit = ' '.join(words[:2])
                    if '-' in order and (len(words) == 4 or len(words) == 5): # move/attack order, 5 for via convoy
                        att_target = words[words.index('-')+1]
                        if att_target not in locs_of_powers: # locatoin controlled by no one
                            filtered_orders.append(order)
                        elif self.attitude[locs_of_powers[att_target]] != 'FRIENDLY':
                            filtered_orders.append(order)
                        else:
                            pass # will not attack friendly country
                    elif ' S ' in order: # support order
                        supported_unit = ' '.join(words[words.index('S')+1:words.index('S')+3])
                        if self.attitude[units_of_powers[supported_unit]] != 'HOSTILE':
                            filtered_orders.append(order)
                        else:
                            pass # will not support hostile country
                    else:
                        pass
                if filtered_orders:
                    power_orders.append(random.choice(filtered_orders)) # random actions from filtered orders
        return power_orders 



class GreedyAgent(Agent):
    '''An agent is greedy with depth-1, without long-term planning'''

    def __init__(self, agent_name='Greedy Agent'):
        super().__init__(agent_name)
        self.map_graph_army = None # this is the connection graph of the map, not the state space graph for the game
        self.map_graph_navy = None # this is the connection graph of the map, not the state space graph for the game

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_map_graphs()

    def build_map_graphs(self):    # You can re-use this function if needed
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(self.game.map.loc_type.keys()) # locations with '/' are not real provinces

        for i in locations:
            if self.game.map.loc_type[i] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(i.upper())
            if self.game.map.loc_type[i] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(i.upper())

        locations = [i.upper() for i in locations]

        for i in locations:
            for j in locations:
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    def get_actions(self):

        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)

        # If not a movement phase, just take random actions
        if self.game.phase_type != 'M':
            power_orders = [random.choice(all_possible_orders[loc]) 
                for loc in orderable_locations if all_possible_orders[loc]]
            return power_orders

        # If it is a movement phase, take greedy actions. That is to move towards enermy centers.
        enermy_centers = []
        for i in self.game.map.scs:
            if i not in self.game.get_centers(self.power_name): # all centers not controlled by self
                enermy_centers.append(i)

        power_orders = []
        for loc in orderable_locations:
            army_possible_orders = [i for i in all_possible_orders[loc] if i[0] == 'A']
            if army_possible_orders:
                if loc in enermy_centers:
                    power_orders.append(f'A {loc} H') # Only after the Fall move do SCs potentially change ownership — that's when the game checks who controls which supply centers.
                else:
                    paths = nx.shortest_path(self.map_graph_army, source=loc)
                    closest_center = None
                    closest_distance = 1000
                    for center in enermy_centers:
                        if center not in paths.keys():
                            continue
                        dis = len(paths[center])
                        if dis < closest_distance:
                            closest_distance = dis
                            closest_center = center
                    if closest_center is not None:
                        assert(len(paths[closest_center]) > 1)
                        target = paths[closest_center][1]
                        power_orders.append(f'A {loc} - {target}')
            
            navy_possible_orders = [i for i in all_possible_orders[loc] if i[0] == 'F']
            if navy_possible_orders:
                if loc in enermy_centers:
                    power_orders.append(f'F {loc} H')
                else:
                    paths = nx.shortest_path(self.map_graph_navy, source=loc)
                    closest_center = None
                    closest_distance = 1000
                    for center in enermy_centers:
                        if center not in paths.keys():
                            continue
                        dis = len(paths[center])
                        if dis < closest_distance:
                            closest_distance = dis
                            closest_center = center
                    if closest_center is not None:
                        assert(len(paths[closest_center]) > 1)
                        target = paths[closest_center][1]
                        power_orders.append(f'F {loc} - {target}')

            if not army_possible_orders and not navy_possible_orders:
                if all_possible_orders[loc]:
                    power_orders.append(random.choice(all_possible_orders[loc]))

        power_orders = list(set(power_orders)) # de-duplicate

        # Support other units if having the same target in an ad-hoc way
        for i in range(len(power_orders)):
            for j in range(i + 1, len(power_orders)):
                if len(power_orders[i].split(' ')) == 4 and len(power_orders[j].split(' ')) == 4: # move/attacks, ignoring 'VIA' convoys, this greedy agent cannot convoy.
                    if power_orders[i].split(' ')[-1] == power_orders[j].split(' ')[-1]: # same target
                        new_order = f'{power_orders[j].split(" ")[0]} {power_orders[j].split(" ")[1]} S {power_orders[i]}' # repalce the second order with a support order
                        power_orders[j] = new_order

        return power_orders


