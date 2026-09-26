import time
import heapq
import random
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
        self.MAX_TARGETS_PER_UNIT = 5

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

    def get_center_owners(self):
        """
        Returns {supply_center_name: power_name}
        """
        owners = {}
        try:
            for p in self.game.powers.keys():
                for c in self.game.get_centers(p):
                    owners[c.upper()] = p
        except Exception:
            pass
        return owners

    def choose_targets(self, loc, kind, friendly_occ, enemy_occ, reserved_dests, center_owners):
        """
        Basic targeting: purely nearest-SC by static graph distance.
        No preference for neutral/enemy/owned status.
        """
        max_targets = getattr(self, "MAX_TARGETS_PER_UNIT", 5)
        dist_table = self.dist_army if kind == "Army" else self.dist_fleet

        if loc not in dist_table:
            return []

        scored = []
        for sc in self.supply_centers:
            if sc in reserved_dests:
                continue
            if sc in friendly_occ:
                continue

            d = dist_table[loc].get(sc, float('inf'))
            if d == float('inf'):
                continue

            scored.append((d, sc))

        scored.sort(key=lambda x: x[0])
        return [sc for _, sc in scored[:max_targets]]

    def add_simple_supports(self, final_orders, unit_options):
        """
        Very simple support logic:
        If one unit is moving to a target and another unit can legally support that move,
        sometimes convert the second unit's move/hold into support.
        """
        order_by_token = {}
        for order in final_orders:
            parts = order.split()
            if len(parts) >= 2:
                token = f"{parts[0]} {parts[1]}"
                order_by_token[token] = order

        move_tokens = []
        for token, order in order_by_token.items():
            parts = order.split()
            if len(parts) >= 4 and parts[2] == '-':
                move_tokens.append((token, order))

        # Prioritize supporting moves into supply centers
        move_tokens.sort(key=lambda x: 0 if x[1].split()[3].upper() in self.supply_centers else 1)

        used_supporters = set()

        for target_token, target_order in move_tokens:
            for support_token, support_order in list(order_by_token.items()):
                if support_token == target_token:
                    continue
                if support_token in used_supporters:
                    continue
                if ' S ' in support_order:
                    continue

                # Find a legal support option for this exact move
                for opt in unit_options.get(support_token, []):
                    if ' S ' in opt and opt.endswith(target_order):
                        order_by_token[support_token] = opt
                        used_supporters.add(support_token)
                        break

                if support_token in used_supporters:
                    break

        return list(order_by_token.values())
    
    def build_static_map(self, game):
        """
        builds adjacency graphs for armies, calculates
        shortest paths to supply centers
        """
        locations_dict = {}
        
        # Uppercase all keys and values to ensure consistency across engine APIs
        if hasattr(game.map, 'loc_type'):
            locations_dict = {k.upper(): str(v).upper() for k, v in game.map.loc_type.items()}
        else:
            raise AttributeError("Cannot find map locations in game.map")

        # Identify Supply Centers
        sc_names = []
        if hasattr(game.map, 'scs'):
            raw_scs = game.map.scs
            for sc in raw_scs:
                name = getattr(sc, 'name', str(sc)).upper()
                if name in locations_dict:
                    sc_names.append(name)
            
        self.supply_centers = sc_names

        # build graphs
        g_army = nx.Graph()
        g_fleet = nx.Graph()
        
        def get_loc_type(loc_name):
            return locations_dict.get(loc_name.upper(), "")

        for loc_name in locations_dict.keys():
            l_type = get_loc_type(loc_name)
            is_sea = 'SEA' in l_type or 'WATER' in l_type
            is_coast = 'COAST' in l_type
            is_land = 'LAND' in l_type
            
            if is_land or is_coast:
                g_army.add_node(loc_name)
            if is_sea or is_coast:
                g_fleet.add_node(loc_name)

        # Use the engine's native abuts() method for highly accurate adjacency mapping
        if hasattr(game.map, 'abuts'):
            locs = list(locations_dict.keys())
            for i in locs:
                for j in locs:
                    if i == j: continue
                    if i in g_army and j in g_army:
                        if game.map.abuts('A', i, '-', j):
                            g_army.add_edge(i, j)
                    if i in g_fleet and j in g_fleet:
                        if game.map.abuts('F', i, '-', j):
                            g_fleet.add_edge(i, j)
        else:
            # Fallback if abuts is somehow not available
            for loc_name in locations_dict.keys():
                neighbors = set()
                if hasattr(game.map, 'adjacent_provinces'):
                    adj = game.map.adjacent_provinces.get(loc_name, [])
                    for n in adj:
                        neighbors.add(getattr(n, 'name', str(n)).upper())
                elif hasattr(game.map, 'get_neighbors'):
                     neighbors = set(n.upper() for n in game.map.get_neighbors(loc_name))
                     
                for nb_name in neighbors:
                    if nb_name not in locations_dict: continue
                    nb_type = get_loc_type(nb_name)
                    if loc_name in g_army and nb_name in g_army:
                        if 'SEA' not in nb_type and 'WATER' not in nb_type:
                            g_army.add_edge(loc_name, nb_name)
                    if loc_name in g_fleet and nb_name in g_fleet:
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
                loc = u.split()[1].upper()
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

                # Do not route through enemy-occupied provinces.
                # Only allow attacking the actual goal if it is enemy-occupied.
                if neighbor in enemy_occ:
                    if neighbor == goal_loc:
                        step_cost += self.ENEMY_PENALTY_ATTACK
                    else:
                        continue

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
                    
        return None

    # step 3. generating actions

    def handle_non_movement(self):
        """
        Handles Retreat and Adjustment phases more safely.
        """
        phase_type = getattr(self.game, 'phase_type', 'M')
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = []

        if phase_type == 'R':
            friendly_occ, enemy_occ = self.get_dynamic_costs()
            center_owners = self.get_center_owners()
            my_centers = {c for c, p in center_owners.items() if p == self.power_name}
        else:
            friendly_occ, enemy_occ = set(), set()
            my_centers = set()

        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue

            strings = [o.as_string() if hasattr(o, 'as_string') else str(o) for o in possible]

            if phase_type == 'R':
                retreats = [s for s in strings if ' R ' in s]
                if retreats:
                    best_retreat = None
                    best_score = float('inf')

                    for s in retreats:
                        parts = s.split()
                        if len(parts) < 4:
                            continue

                        dest = parts[3].upper()
                        unit_kind = "Fleet" if parts[0] == "F" else "Army"
                        dist_table = self.dist_fleet if unit_kind == "Fleet" else self.dist_army

                        score = 0.0

                        if dest in enemy_occ:
                            score += 1000.0
                        if dest in friendly_occ:
                            score += 500.0

                        # Prefer retreating closer to our owned centers
                        if dest in dist_table and my_centers:
                            dists = [dist_table[dest].get(c, float('inf')) for c in my_centers]
                            best_dist = min(dists)
                            if best_dist != float('inf'):
                                score += best_dist

                        if score < best_score:
                            best_score = score
                            best_retreat = s

                    power_orders.append(best_retreat if best_retreat else random.choice(retreats))
                else:
                    disbands = [s for s in strings if s.split() and s.split()[-1] == 'D']
                    power_orders.append(disbands[0] if disbands else random.choice(strings))

            elif phase_type == 'A':
                builds = [s for s in strings if s.split() and s.split()[-1] == 'B']
                if builds:
                    armies = [s for s in builds if s.startswith('A ')]
                    fleets = [s for s in builds if s.startswith('F ')]

                    # Simple heuristic: prefer armies unless only fleets are available.
                    # You can improve this later based on map position.
                    power_orders.append(random.choice(armies) if armies else random.choice(fleets))
                else:
                    disbands = [s for s in strings if s.split() and s.split()[-1] == 'D']
                    if disbands:
                        power_orders.append(disbands[0])
                    else:
                        power_orders.append(random.choice(strings))
            else:
                power_orders.append(random.choice(strings))

        return power_orders
    
    def get_actions(self):
        """
        Uses A* to plan routes and take actions on first steps.
        Improved target selection and simple support logic.
        """
        deadline = time.perf_counter() + self.TIME_BUDGET

        # Use phase_type directly to accurately detect non-movement phases
        if getattr(self.game, 'phase_type', 'M') != 'M':
            return self.handle_non_movement()

        # Move Phase Logic
        try:
            all_possible_orders = self.game.get_all_possible_orders()
        except Exception:
            return []

        orderable_locations = self.game.get_orderable_locations(self.power_name)

        # orders grouped by unit token
        unit_options = defaultdict(list)
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            for o in possible:
                order_str = o.as_string() if hasattr(o, 'as_string') else str(o)
                parts = order_str.split()
                if len(parts) >= 2:
                    token = f"{parts[0]} {parts[1]}"
                    unit_options[token].append(order_str)

        if not unit_options:
            return []

        friendly_occ, enemy_occ = self.get_dynamic_costs()
        center_owners = self.get_center_owners()

        final_orders = []
        reserved_dests = set()

        # sort units by those with the fewest options first
        sorted_tokens = sorted(unit_options.keys(), key=lambda t: len(unit_options[t]))

        for token in sorted_tokens:
            options = unit_options[token]
            hold_opt = next((opt for opt in options if opt.endswith(' H')), None)

            if time.perf_counter() > deadline:
                final_orders.append(hold_opt if hold_opt else options[0])
                continue

            prefix, loc = token.split()
            loc = loc.upper()
            kind = "Fleet" if prefix == "F" else "Army"

            dist_table = self.dist_army if kind == "Army" else self.dist_fleet
            if loc not in dist_table:
                final_orders.append(hold_opt if hold_opt else options[0])
                continue

            # Choose meaningful targets: neutrals / enemy SCs, not our own centers if possible
            top_targets = self.choose_targets(
                loc,
                kind,
                friendly_occ,
                enemy_occ,
                reserved_dests,
                center_owners
            )

            chosen_next_step = None
            chosen_target = None

            # run A* to each target
            for target_sc in top_targets:
                if time.perf_counter() > deadline:
                    break

                path = self.astar_search(loc, target_sc, kind, friendly_occ, enemy_occ, deadline)

                if path and len(path) > 1:
                    next_step = path[1]

                    # Check if this move is actually legal according to engine
                    matching_option = None
                    for opt in options:
                        parts_opt = opt.split()
                        if len(parts_opt) >= 4 and parts_opt[2] == '-':
                            if parts_opt[3].upper() == next_step:
                                matching_option = opt
                                break

                    if matching_option:
                        chosen_next_step = matching_option
                        chosen_target = target_sc
                        break

            # Execute Best Move or Fallback
            if chosen_next_step:
                final_orders.append(chosen_next_step)

                dest = chosen_next_step.split()[3].upper()
                reserved_dests.add(dest)

                # Also reserve the strategic target so multiple units do not aim for the same SC
                if chosen_target:
                    reserved_dests.add(chosen_target)

                # Update local planning occupancy to reduce self-collision
                friendly_occ.discard(loc)
                friendly_occ.add(dest)

            else:
                # fallback: greedy one-step move
                best_order = None
                best_score = -float('inf')

                for opt in options:
                    parts = opt.split()
                    if len(parts) >= 4 and parts[2] == '-':
                        dest = parts[3].upper()

                        if dest in reserved_dests:
                            continue
                        if dest in friendly_occ:
                            continue

                        curr_min = min(dist_table[loc].values()) if dist_table[loc] else float('inf')
                        dest_min = min(dist_table[dest].values()) if dest in dist_table and dist_table[dest] else float('inf')

                        score = curr_min - dest_min

                        if dest in self.supply_centers:
                            score += 50

                        if dest in enemy_occ:
                            score -= 5

                        if score > best_score:
                            best_score = score
                            best_order = opt

                if best_order:
                    final_orders.append(best_order)
                    dest = best_order.split()[3].upper()
                    reserved_dests.add(dest)
                    friendly_occ.discard(loc)
                    friendly_occ.add(dest)
                else:
                    final_orders.append(hold_opt if hold_opt else options[0])

        # Add simple support orders where possible
        final_orders = self.add_simple_supports(final_orders, unit_options)

        return final_orders