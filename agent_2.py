import time
import heapq
import networkx as nx
from collections import defaultdict
from agent_baselines import Agent

class StudentAgent(Agent):
    """
    A* path planning accounting for certain risk
    
    Strategy:
    1. build map graph and compute distances to all supply centres
    2. for every unit, identify possible promising target supply centres
    3. Run A* search from current spot to target supply centres
        includes penalties for moving near or through enemies
        heuristic is the static distance to the target SC
    4. execute the first step of best found path
    5. replan in next phase
    """

    def __init__(self, agent_name="AStarAgent"):
        super().__init__(agent_name)
        self.game = None
        self.power_name = None
        
        # static data
        self.graph_army = None
        self.graph_fleet = None
        self.dist_army = {}
        self.dist_fleet = {}
        self.supply_centers = []

        # constants (for tuning the heuristics)
        self.TIME_BUDGET = 0.85     # max time for get_actions()
        self.MAX_NODES_PER_UNIT = 300 # limit search size
        self.ENEMY_PENALTY_PASS = 15.0 # cost to move past enemy province
        self.ENEMY_PENALTY_ATTACK = 5.0 # cost to attack an enemy province
        self.FRIENDLY_BLOCK_COST = float('inf') # cannot make routes through own units

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_static_map(game)

    def update_game(self, all_power_orders):
        # do not change this code
        for p in all_power_orders.keys():
            self.game.set_orders(p, all_power_orders[p])
        self.game.process()

    # step 1: building the map

    def build_static_map(self, game):
        """
        builds adjacency graphs for armies, calculates
        shortest paths to supply centers
        """
        locations_dict = {}
        
        # Try standard API: game.map.loc_type
        if hasattr(game.map, 'loc_type'):
            locations_dict = game.map.loc_type.copy()
        else:
            raise AttributeError("Cannot find map locations in game.map")

        # Identify Supply Centers
        sc_names = []
        if hasattr(game.map, 'scs'):
            raw_scs = game.map.scs
            for sc in raw_scs:
                name = getattr(sc, 'name', str(sc))
                if name in locations_dict:
                    sc_names.append(name)
            
        self.supply_centers = sc_names

        # build graphs
        g_army = nx.Graph()
        g_fleet = nx.Graph()
        
        def get_loc_type(loc_name):
            return str(locations_dict.get(loc_name, "")).upper()

        for loc_name in locations_dict.keys():
            l_type = get_loc_type(loc_name)
            is_sea = 'SEA' in l_type or 'WATER' in l_type
            is_coast = 'COAST' in l_type
            is_land = 'LAND' in l_type
            
            # check army type (land or coast)
            if is_land or is_coast:
                g_army.add_node(loc_name)
                
                # Get neighbors
                neighbors = set()
                if hasattr(game.map, 'adjacent_provinces'):
                    adj = game.map.adjacent_provinces.get(loc_name, [])
                    for n in adj:
                        neighbors.add(getattr(n, 'name', str(n)))
                elif hasattr(game.map, 'get_neighbors'):
                     neighbors = set(game.map.get_neighbors(loc_name))
                
                for nb_name in neighbors:
                    if nb_name not in locations_dict: continue
                    nb_type = get_loc_type(nb_name)
                    # Armies cannot enter Sea/Water
                    if 'SEA' not in nb_type and 'WATER' not in nb_type:
                        g_army.add_edge(loc_name, nb_name)

            # FLEETS: Sea or Coast
            if is_sea or is_coast:
                g_fleet.add_node(loc_name)
                
                neighbors = set()
                if hasattr(game.map, 'adjacent_provinces'):
                    adj = game.map.adjacent_provinces.get(loc_name, [])
                    for n in adj:
                        neighbors.add(getattr(n, 'name', str(n)))
                elif hasattr(game.map, 'get_neighbors'):
                     neighbors = set(game.map.get_neighbors(loc_name))
                     
                for nb_name in neighbors:
                    if nb_name not in locations_dict: continue
                    nb_type = get_loc_type(nb_name)
                    # Fleets cannot enter Land
                    if 'SEA' in nb_type or 'WATER' in nb_type or 'COAST' in nb_type:
                        g_fleet.add_edge(loc_name, nb_name)

        self.graph_army = g_army
        self.graph_fleet = g_fleet

        # precompute distances
        def compute_dist_table(graph, sc_list):
            table = {}
            for node in graph.nodes():
                lengths = nx.single_source_shortest_path_length(graph, node)
                table[node] = {sc: lengths.get(sc, float('inf')) for sc in sc_list}
            return table

        self.dist_army = compute_dist_table(g_army, self.supply_centers)
        self.dist_fleet = compute_dist_table(g_fleet, self.supply_centers)
        
        print(f"A* Agent Initialized. SCs: {len(self.supply_centers)}, Army Nodes: {g_army.number_of_nodes()}")

    # step 2. A* search

    def get_dynamic_costs(self):
        """
        returns sets of occupied provinces for penalty calculation
        """
        friendly_occ = set()
        enemy_occ = set()
        
        for pname, power in self.game.powers.items():
            for u in power.units:
                # Units are formatted like "A LON"
                loc = u.split()[1]
                if pname == self.power_name:
                    friendly_occ.add(loc)
                else:
                    enemy_occ.add(loc)
                    
        return friendly_occ, enemy_occ

    def astar_search(self, start_loc, goal_loc, unit_kind, friendly_occ, enemy_occ, deadline):
        """
        Performs A* search from start_loc to goal_loc.
        Returns list of provinces representing the path, or None if failed.
        """
        if start_loc == goal_loc:
            return [start_loc]
            
        # Select appropriate graph and distance table
        if unit_kind == "Army":
            adj_graph = self.graph_army
            dist_table = self.dist_army
        else:
            adj_graph = self.graph_fleet
            dist_table = self.dist_fleet
            
        if start_loc not in adj_graph or goal_loc not in dist_table.get(start_loc, {}):
            # cannot be reached
            return None 

        start_h = dist_table[start_loc].get(goal_loc, float('inf'))
        if start_h == float('inf'):
            return None
            
        open_heap = [(start_h, 0.0, start_loc)]
        g_scores = {start_loc: 0.0}
        parents = {}
        closed_set = set()
        
        expanded_count = 0

        while open_heap:
            # check if time or node limits are exceeded
            if time.perf_counter() > deadline or expanded_count > self.MAX_NODES_PER_UNIT:
                return None
                
            f, g, current = heapq.heappop(open_heap)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            expanded_count += 1
            
            if current == goal_loc:
                # reconstruct path
                path = [current]
                while current in parents:
                    current = parents[current]
                    path.append(current)
                return path[::-1]
                
            for neighbor in adj_graph.neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                # calculate base movement cost
                step_cost = 1.0 
                
                # Cannot plan through own units
                if neighbor in friendly_occ and neighbor != goal_loc:
                    continue 
                    
                # Enemy Penalty
                if neighbor in enemy_occ:
                    if neighbor == goal_loc:
                        # attacking target is allowed but has penalty
                        step_cost += self.ENEMY_PENALTY_ATTACK
                    else:
                        # passing through enemy territory is expensiveA
                        step_cost += self.ENEMY_PENALTY_PASS
                
                tentative_g = g + step_cost
                
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    parents[neighbor] = current
                    
                    # Static distance to goal
                    h = dist_table[neighbor].get(goal_loc, float('inf'))
                    if h == float('inf'):
                        continue
                        
                    f_new = tentative_g + h
                    heapq.heappush(open_heap, (f_new, tentative_g, neighbor))
                    
        # no path located
        return None

    # step 3. generating actions

    def get_legal_orders(self):
        """
        extract legal orders for this power.
        """
        legal_strings = []
        try:
            all_orders_dict = self.game.get_all_possible_orders()
        except Exception:
            return []

        my_units_str = self.game.powers[self.power_name].units
        
        for unit_str in my_units_str:
            parts = unit_str.split()
            if len(parts) < 2: 
                continue
            loc_code = parts[1]
            
            if loc_code in all_orders_dict:
                raw_orders = all_orders_dict[loc_code]
                for o in raw_orders:
                    s = o.as_string() if hasattr(o, 'as_string') else str(o)
                    legal_strings.append(s)
        return legal_strings

    def get_actions(self):
        """
        uses A* to plan routes and take actions on first steps
        """
        deadline = time.perf_counter() + self.TIME_BUDGET
        phase = self.game.get_current_phase()
        
        # check which phase the game is in
        if "Retreat" in phase:
            return self.handle_retreats(deadline)
        if "Adjustment" in phase:
            return self.handle_adjustments(deadline)

        # Move Phase Logic
        legal_order_strings = self.get_legal_orders()
        if not legal_order_strings:
            return []

        # orders grouped by unit token
        unit_options = defaultdict(list)
        for order_str in legal_order_strings:
            parts = order_str.split()
            if len(parts) >= 2:
                token = f"{parts[0]} {parts[1]}"
                unit_options[token].append(order_str)

        friendly_occ, enemy_occ = self.get_dynamic_costs()
        final_orders = []
        reserved_dests = set()

        # sort units by those with the fewest options first
        # so that other units with more options dont take their moves
        sorted_tokens = sorted(unit_options.keys(), key=lambda t: len(unit_options[t]))

        for token in sorted_tokens:
            if time.perf_counter() > deadline:
                break
                
            options = unit_options[token]
            prefix, loc = token.split()
            kind = "Fleet" if prefix == "F" else "Army"
            
            # Identify Candidate Targets (Unowned SCs)
            
            dist_table = self.dist_army if kind == "Army" else self.dist_fleet
            if loc not in dist_table:
                 final_orders.append(f"{token} H")
                 continue

            # get all reachable supply centres sorted by static distance
            candidate_scs = [sc for sc, d in dist_table[loc].items() if d != float('inf')]
            candidate_scs.sort(key=lambda sc: dist_table[loc][sc])
            
            # take top 3 targets to limit A* searches
            top_targets = candidate_scs[:3]
            
            chosen_next_step = None
            
            # run A* to each target
            for target_sc in top_targets:
                if time.perf_counter() > deadline:
                    break
                    
                # dont target a supply centre already reserved by another unit this turn
                if target_sc in reserved_dests:
                    continue

                path = self.astar_search(loc, target_sc, kind, friendly_occ, enemy_occ, deadline)
                
                if path and len(path) > 1:
                    next_step = path[1]
                    
                    # Check if this move is actually legal according to engine
                    matching_option = None
                    for opt in options:
                        parts_opt = opt.split()
                        if len(parts_opt) >= 4 and parts_opt[2] == '-':
                            if parts_opt[3] == next_step:
                                matching_option = opt
                                break
                    
                    if matching_option:
                        chosen_next_step = matching_option
                        # valid path found
                        break

            # 3. Execute Best Move or Fallback
            if chosen_next_step:
                final_orders.append(chosen_next_step)
                # Reserve destination
                dest = chosen_next_step.split()[3]
                reserved_dests.add(dest)
                
                # Update local planning occupancy to reduce self-collision
                friendly_occ.discard(loc)
                friendly_occ.add(dest)
            else:
                # fallback is to use greedy distance
                best_order = None
                best_score = -float('inf')
                
                for opt in options:
                    parts = opt.split()
                    if len(parts) >= 4 and parts[2] == '-':
                        dest = parts[3]
                        if dest in reserved_dests:
                            continue
                        
                        # score establishes that closer to supply center is better
                        curr_min = min(dist_table[loc].values()) if dist_table[loc] else float('inf')
                        dest_min = min(dist_table[dest].values()) if dest in dist_table and dist_table[dest] else float('inf')
                        
                        score = curr_min - dest_min
                        if dest in self.supply_centers:
                            score += 50
                            
                        if score > best_score:
                            best_score = score
                            best_order = opt
                
                if best_order:
                    final_orders.append(best_order)
                    dest = best_order.split()[3]
                    reserved_dests.add(dest)
                    friendly_occ.discard(loc)
                    friendly_occ.add(dest)
                else:
                    final_orders.append(f"{token} H")

        return final_orders

    def handle_retreats(self, deadline):
        """retreat logic is to move to safest nearby province."""
        _, legal_retreats, disbands = self.parse_orders()
        tokens = set(legal_retreats.keys()) | disbands
        if not tokens:
            return []
            
        orders = []
        reserved = set()
        friendly_occ, enemy_occ = self.get_dynamic_costs()
        
        for token in sorted(tokens):
            prefix, loc = token.split()
            kind = "Fleet" if prefix == "F" else "Army"
            legal = set(legal_retreats.get(token, ())) - reserved
            
            best_dest = None
            best_score = float('inf')
            
            dist_table = self.dist_army if kind == "Army" else self.dist_fleet
            
            for dest in legal:
                # Prefer retreating towards own SCs or away from enemies
                score = 0
                if dest in enemy_occ:
                    score += 100 # high penalty
                if dest in friendly_occ:
                    score += 50 # medium penalty due to risk of collision
                    
                # distance to nearest SC
                if dest in dist_table and dist_table[dest]:
                    min_d = min(dist_table[dest].values())
                    score += min_d * 2
                
                if score < best_score:
                    best_score = score
                    best_dest = dest
            
            if best_dest:
                orders.append(f"{token} R {best_dest}")
                reserved.add(best_dest)
            elif token in disbands:
                orders.append(f"{token} D")
            else:
                # Force disband if no safe retreat
                orders.append(f"{token} D") 
                
        return orders

    def handle_adjustments(self, deadline):
        # currently a placeholder, returning empty list, for winter adjustment phase
        return []

    def parse_orders(self):
        """for parsing legal order into dictionaries"""
        moves = defaultdict(set)
        retreats = defaultdict(set)
        disbands = set()
        
        try:
            all_orders_dict = self.game.get_all_possible_orders()
        except Exception:
            return moves, retreats, disbands

        my_units_str = self.game.powers[self.power_name].units
        
        for unit_str in my_units_str:
            parts = unit_str.split()
            if len(parts) < 2: continue
            loc_code = parts[1]
            
            if loc_code in all_orders_dict:
                raw_orders = all_orders_dict[loc_code]
                for o in raw_orders:
                    s = o.as_string() if hasattr(o, 'as_string') else str(o)
                    sparts = s.split()
                    if len(sparts) < 3: continue
                    
                    token = f"{sparts[0]} {sparts[1]}"
                    action = sparts[2]
                    
                    if action == '-':
                        moves[token].add(sparts[3])
                    elif action == 'R':
                        retreats[token].add(sparts[3])
                    elif action == 'D':
                        disbands.add(token)
                        
        return dict(moves), dict(retreats), disbands