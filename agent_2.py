import time
import heapq
import random
import networkx as nx
from collections import defaultdict
from agent_baselines import Agent


class StudentAgent(Agent):
    """
    Basic A* agent plus ONE major new technique:

    Context-Aware Targeting:
    - Prefer neutral supply centers.
    - Prefer enemy supply centers.
    - Avoid targeting our own supply centers unless necessary.
    """

    def __init__(self, agent_name="AStarAgent"):
        super().__init__(agent_name)

        self.game = None
        self.power_name = None

        # Static map data
        self.graph_army = None
        self.graph_fleet = None
        self.dist_army = {}
        self.dist_fleet = {}
        self.supply_centers = []

        # Constants
        self.TIME_BUDGET = 0.85
        self.MAX_NODES_PER_UNIT = 300
        self.ENEMY_PENALTY_PASS = 15.0
        self.ENEMY_PENALTY_ATTACK = 5.0
        self.FRIENDLY_BLOCK_COST = float("inf")

        # Number of A* targets to try per unit
        self.MAX_TARGETS_PER_UNIT = 3

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_static_map(game)

    def update_game(self, all_power_orders):
        # Do not change this code.
        for p in all_power_orders.keys():
            self.game.set_orders(p, all_power_orders[p])
        self.game.process()

    ###########################################################################
    # Static map building
    ###########################################################################

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
        Builds army/fleet graphs and precomputes distances to supply centers.
        """
        locations_dict = {}

        if hasattr(game.map, "loc_type"):
            locations_dict = {
                str(k).upper(): str(v).upper()
                for k, v in game.map.loc_type.items()
            }
        else:
            raise AttributeError("Cannot find map locations in game.map")

        # Identify supply centers.
        sc_set = set()
        if hasattr(game.map, "scs"):
            for sc in game.map.scs:
                name = str(getattr(sc, "name", sc)).upper()

                if name in locations_dict:
                    sc_set.add(name)

                base = name.split("/")[0]
                if base in locations_dict:
                    sc_set.add(base)

        self.supply_centers = sorted(sc_set)

        g_army = nx.Graph()
        g_fleet = nx.Graph()

        for loc_name, loc_type in locations_dict.items():
            is_sea = "SEA" in loc_type or "WATER" in loc_type
            is_coast = "COAST" in loc_type
            is_land = "LAND" in loc_type

            if is_land or is_coast:
                g_army.add_node(loc_name)

            if is_sea or is_coast:
                g_fleet.add_node(loc_name)

        # Use the engine's adjacency checker if available.
        if hasattr(game.map, "abuts"):
            locs = list(locations_dict.keys())

            for i in locs:
                for j in locs:
                    if i == j:
                        continue

                    try:
                        if i in g_army and j in g_army:
                            if game.map.abuts("A", i, "-", j):
                                g_army.add_edge(i, j)

                        if i in g_fleet and j in g_fleet:
                            if game.map.abuts("F", i, "-", j):
                                g_fleet.add_edge(i, j)
                    except Exception:
                        pass

        else:
            # Fallback neighbor construction.
            for loc_name in locations_dict.keys():
                neighbors = set()

                if hasattr(game.map, "adjacent_provinces"):
                    adj = game.map.adjacent_provinces.get(loc_name, [])
                    for n in adj:
                        neighbors.add(str(getattr(n, "name", n)).upper())

                elif hasattr(game.map, "get_neighbors"):
                    neighbors = set(
                        str(n).upper() for n in game.map.get_neighbors(loc_name)
                    )

                for nb in neighbors:
                    if nb not in locations_dict:
                        continue

                    nb_type = locations_dict[nb]

                    if loc_name in g_army and nb in g_army:
                        if "SEA" not in nb_type and "WATER" not in nb_type:
                            g_army.add_edge(loc_name, nb)

                    if loc_name in g_fleet and nb in g_fleet:
                        if (
                            "SEA" in nb_type
                            or "WATER" in nb_type
                            or "COAST" in nb_type
                        ):
                            g_fleet.add_edge(loc_name, nb)

        self.graph_army = g_army
        self.graph_fleet = g_fleet

        def compute_dist_table(graph, sc_list):
            table = {}
            for node in graph.nodes():
                lengths = nx.single_source_shortest_path_length(graph, node)
                table[node] = {
                    sc: lengths.get(sc, float("inf"))
                    for sc in sc_list
                }
            return table

        self.dist_army = compute_dist_table(g_army, self.supply_centers)
        self.dist_fleet = compute_dist_table(g_fleet, self.supply_centers)

        print(
            f"A* Agent Initialized. "
            f"SCs: {len(self.supply_centers)}, "
            f"Army Nodes: {g_army.number_of_nodes()}"
        )

    ###########################################################################
    # Dynamic helpers
    ###########################################################################

    def get_dynamic_costs(self):
        """
        Returns sets of friendly-occupied and enemy-occupied provinces.
        """
        friendly_occ = set()
        enemy_occ = set()

        try:
            for pname, power in self.game.powers.items():
                for u in power.units:
                    parts = u.split()
                    if len(parts) < 2:
                        continue

                    loc = parts[1].upper()

                    if pname == self.power_name:
                        friendly_occ.add(loc)
                    else:
                        enemy_occ.add(loc)
        except Exception:
            pass

        return friendly_occ, enemy_occ

    def get_center_owners(self):
        """
        NEW TECHNIQUE HELPER:

        Returns:
            {supply_center_name: power_name}

        This allows the agent to avoid targeting its own centers and to
        prefer neutral/enemy centers.
        """
        owners = {}

        try:
            for p in self.game.powers.keys():
                for c in self.game.get_centers(p):
                    owners[str(c).upper()] = p
        except Exception:
            pass

        return owners

    ###########################################################################
    # A* search
    ###########################################################################

    def astar_search(
        self,
        start_loc,
        goal_loc,
        unit_kind,
        friendly_occ,
        enemy_occ,
        deadline
    ):
        """
        Basic A* search from start_loc to goal_loc.
        Returns a path as a list of locations, or None.
        """
        start_loc = str(start_loc).upper()
        goal_loc = str(goal_loc).upper()

        if start_loc == goal_loc:
            return [start_loc]

        if unit_kind == "Army":
            adj_graph = self.graph_army
            dist_table = self.dist_army
        else:
            adj_graph = self.graph_fleet
            dist_table = self.dist_fleet

        if start_loc not in adj_graph:
            return None

        if goal_loc not in dist_table.get(start_loc, {}):
            return None

        start_h = dist_table[start_loc].get(goal_loc, float("inf"))
        if start_h == float("inf"):
            return None

        open_heap = [(start_h, 0.0, start_loc)]
        g_scores = {start_loc: 0.0}
        parents = {}
        closed_set = set()
        expanded_count = 0

        while open_heap:
            if time.perf_counter() > deadline:
                return None

            if expanded_count > self.MAX_NODES_PER_UNIT:
                return None

            f, g, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            closed_set.add(current)
            expanded_count += 1

            if current == goal_loc:
                path = [current]

                while current in parents:
                    current = parents[current]
                    path.append(current)

                return path[::-1]

            for neighbor in adj_graph.neighbors(current):
                if neighbor in closed_set:
                    continue

                step_cost = 1.0

                # Cannot plan through own units.
                if neighbor in friendly_occ and neighbor != goal_loc:
                    continue

                # Enemy penalty.
                if neighbor in enemy_occ:
                    if neighbor == goal_loc:
                        step_cost += self.ENEMY_PENALTY_ATTACK
                    else:
                        step_cost += self.ENEMY_PENALTY_PASS

                tentative_g = g + step_cost

                if tentative_g < g_scores.get(neighbor, float("inf")):
                    g_scores[neighbor] = tentative_g
                    parents[neighbor] = current

                    h = dist_table[neighbor].get(goal_loc, float("inf"))
                    if h == float("inf"):
                        continue

                    f_new = tentative_g + h
                    heapq.heappush(open_heap, (f_new, tentative_g, neighbor))

        return None

    ###########################################################################
    # Non-movement fallback
    ###########################################################################

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

    ###########################################################################
    # Main action generation
    ###########################################################################

    def get_actions(self):
        """
        Movement-phase logic.

        The only major new technique here is Context-Aware Targeting.
        """
        deadline = time.perf_counter() + self.TIME_BUDGET

        # Basic phase check.
        if getattr(self.game, "phase_type", "M") != "M":
            return self.handle_non_movement()

        try:
            all_possible_orders = self.game.get_all_possible_orders()
            orderable_locations = self.game.get_orderable_locations(self.power_name)
        except Exception:
            return []

        # Group orders by unit token, e.g. "A LON".
        unit_options = defaultdict(list)

        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])

            for order in possible:
                order_str = order.as_string() if hasattr(order, "as_string") else str(order)
                parts = order_str.split()

                if len(parts) >= 2:
                    token = f"{parts[0]} {parts[1]}"
                    unit_options[token].append(order_str)

        if not unit_options:
            return []

        friendly_occ, enemy_occ = self.get_dynamic_costs()

        ########################################################################
        # NEW TECHNIQUE:
        # Get current supply-center ownership so we can avoid targeting
        # our own centers and prefer neutral/enemy centers.
        ########################################################################
        center_owners = self.get_center_owners()

        final_orders = []
        reserved_dests = set()

        # Process units with fewer legal options first.
        sorted_tokens = sorted(
            unit_options.keys(),
            key=lambda t: len(unit_options[t])
        )

        for token in sorted_tokens:
            options = unit_options[token]

            hold_opt = None
            for opt in options:
                parts = opt.split()
                if parts and parts[-1].upper() == "H":
                    hold_opt = opt
                    break

            if time.perf_counter() > deadline:
                final_orders.append(hold_opt if hold_opt else options[0])
                continue

            token_parts = token.split()
            if len(token_parts) < 2:
                final_orders.append(hold_opt if hold_opt else options[0])
                continue

            prefix = token_parts[0]
            loc = token_parts[1].upper()
            kind = "Fleet" if prefix == "F" else "Army"

            dist_table = self.dist_army if kind == "Army" else self.dist_fleet

            if loc not in dist_table:
                final_orders.append(hold_opt if hold_opt else options[0])
                continue

            ####################################################################
            # NEW TECHNIQUE:
            # Context-Aware Targeting
            #
            # Old behavior:
            #   target the nearest supply centers.
            #
            # New behavior:
            #   target supply centers not owned by us.
            #   Slightly prefer neutral centers.
            #   Penalize enemy-occupied centers.
            ####################################################################
            scored_targets = []

            for sc, d in dist_table[loc].items():
                if d == float("inf"):
                    continue

                if sc in friendly_occ:
                    continue

                owner = center_owners.get(sc)

                # Only consider centers not owned by us.
                if owner != self.power_name:
                    score = float(d)

                    # Prefer neutral centers slightly.
                    if owner is None:
                        score -= 1.0

                    # Enemy-occupied centers are riskier.
                    if sc in enemy_occ:
                        score += 5.0

                    scored_targets.append((score, sc))

            # If for some reason all centers are owned by us, fall back to
            # the original behavior.
            if not scored_targets:
                for sc, d in dist_table[loc].items():
                    if d == float("inf"):
                        continue

                    if sc in friendly_occ:
                        continue

                    scored_targets.append((float(d), sc))

            scored_targets.sort(key=lambda x: x[0])

            top_targets = [
                sc for _, sc in scored_targets[:self.MAX_TARGETS_PER_UNIT]
            ]

            chosen_next_step = None

            # Try A* toward each selected target.
            for target_sc in top_targets:
                if time.perf_counter() > deadline:
                    break

                if target_sc in reserved_dests:
                    continue

                path = self.astar_search(
                    loc,
                    target_sc,
                    kind,
                    friendly_occ,
                    enemy_occ,
                    deadline
                )

                if path and len(path) > 1:
                    next_step = path[1].upper()

                    matching_option = None

                    for opt in options:
                        opt_parts = opt.split()

                        if len(opt_parts) >= 4 and opt_parts[2] == "-":
                            if opt_parts[3].upper() == next_step:
                                matching_option = opt
                                break

                    if matching_option:
                        chosen_next_step = matching_option
                        break

            if chosen_next_step:
                final_orders.append(chosen_next_step)

                dest = chosen_next_step.split()[3].upper()
                reserved_dests.add(dest)

                # Update local planning occupancy.
                friendly_occ.discard(loc)
                friendly_occ.add(dest)

            else:
                # Basic greedy fallback.
                best_order = None
                best_score = -float("inf")

                for opt in options:
                    parts = opt.split()

                    if len(parts) >= 4 and parts[2] == "-":
                        dest = parts[3].upper()

                        if dest in reserved_dests:
                            continue

                        if dest in friendly_occ:
                            continue

                        curr_min = (
                            min(dist_table[loc].values())
                            if dist_table.get(loc)
                            else float("inf")
                        )

                        dest_min = (
                            min(dist_table[dest].values())
                            if dest in dist_table and dist_table.get(dest)
                            else float("inf")
                        )

                        score = curr_min - dest_min

                        if dest in self.supply_centers:
                            score += 50

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

        final_orders = self.add_simple_supports(final_orders, unit_options)

        return final_orders