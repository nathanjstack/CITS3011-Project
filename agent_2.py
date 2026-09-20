import heapq
import time
from collections import defaultdict
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

    def __init__(self, agent_name="LegalMover"):
        super().__init__(agent_name)
        self.game = None
        self.power_name = None

    # @timeout_decorator.timeout(1)  # Commented out for Windows compatibility
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

    def update_game(self, all_power_orders):
        # do not change this code
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    def get_legal_orders(self):
        """
        Manually extracts legal orders for this power using get_all_possible_orders().
        Returns a list of order STRINGS.
        """
        legal_strings = []
        
        # get all possible orders as a dictionary ({province_code: [order_objects]})
        try:
            all_orders_dict = self.game.get_all_possible_orders()
        except Exception as e:
            print(f"Error getting all orders: {e}")
            return []

        # identify current locations of units
        # units are formatted as strings such as  "A LON" or "F NTH"
        my_units_str = self.game.powers[self.power_name].units
        
        for unit_str in my_units_str:
            # parse string formats (e.g A LON to LON)
            parts = unit_str.split()
            if len(parts) < 2:
                continue
            
            loc_code = parts[1]
            
            # check orders for this area
            if loc_code in all_orders_dict:
                raw_orders = all_orders_dict[loc_code]
                
                for o in raw_orders:
                    # Convert Order Object to String
                    if hasattr(o, 'as_string'):
                        legal_strings.append(o.as_string())
                    else:
                        legal_strings.append(str(o))
                        
        return legal_strings

    def get_actions(self):
        """
        Main entry point.
        """
        try:
            legal_order_strings = self.get_legal_orders()

            # if cant find any legal orders return empty lists
            if not legal_order_strings:
                return []

            unit_choices = {}
            
            for order_str in legal_order_strings:
                parts = order_str.split()
                if len(parts) < 2:
                    continue
                
                unit_token = f"{parts[0]} {parts[1]}"
                
                # Store the last option seen for this unit (which i believe may be attack)
                unit_choices[unit_token] = order_str

            final_orders = list(unit_choices.values())
            return final_orders

        except Exception as e:
            print(f"ERROR in get_actions: {e}")
            import traceback
            traceback.print_exc()
            return []