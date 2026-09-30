"""
Group 2 - CITS3011 Diplomacy agent.

Only StudentAgent (the final agent: Technique 3 enabled) is used by the game.
The other classes are the earlier stages reported in the report.

    Basic technique : BasicAStarAgent   (per-unit A*, nearest-SC targeting)
    Technique 1     : Technique1Agent   (context-aware target selection)
    Technique 2     : Technique2Agent   (joint-order planning
    Technique 3     : StudentAgent      (opponent modelling + best response)
"""
import time
import heapq
import random
import networkx as nx
from collections import defaultdict
from agent_baselines import Agent

INF = float("inf")


class StudentAgent(Agent):
    """
    Search-based Diplomacy agent.

    Pipeline per movement phase:
      goal assignment -> enemy scenario model (hold / greedy / coordinated
      attack) -> internal adjudicator -> local search over joint orders
      (attack+support and convoy macros) -> ENEMY BEST RESPONSE to our plan
      (only for opponents that look competent) -> re-search against the mix.

    The best-response step is what makes the agent robust against strong
    opponents (Hidden Agent in scenario 3, other groups' agents in scenario 4).
    Against passive/weak opponents it is skipped, so scenario 1/2 behaviour is
    unchanged.
    """

    def __init__(self, agent_name="SearchAgent", opponent_model=True):
        super().__init__(agent_name)
        # Technique 3 switch. True = final agent. False = Technique 2 only:
        # enemies are assumed to hold, no history observation, no best response.
        self.opponent_model = opponent_model
        self.game = None
        self.power_name = None
        self.graph_army = None
        self.graph_fleet = None
        self.dist_army = {}
        self.dist_fleet = {}
        self.all_dist_army = {}
        self.all_dist_fleet = {}
        self.supply_centers = []
        self.static_order_streak = {}
        self.static_scenario = False
        self.last_enemy_targets = set()
        self.hostile_powers = set()
        self.home_centers = set()

        self.SEARCH_BUDGET = 0.50       # seconds per movement phase (upper bound)
        self.MAX_PASSES = 4
        self.MAX_RESTARTS = 12
        self.FALL_CAPTURE = 10.0
        self.SPRING_CAPTURE = 3.5
        self.UNIT_LOSS = 6.0
        self.SC_LOSS_FALL = 9.0
        self.SC_LOSS_SPRING = 3.0
        self.PROGRESS = 1.2
        self.DISLODGE_BONUS = 2.5
        self.BUILD_ROOM_BONUS = 2.0
        self.SCENARIO_WEIGHTS = (0.3, 0.4, 0.3)   # hold, greedy, attack

        # --- new knobs -------------------------------------------------------
        self.BR_WEIGHT = 0.30           # weight of the enemy best-response scenario (0 disables)
        self.BR_TIME_SHARE = 0.15       # share of the budget spent computing the best response
        self.PHASE1_SHARE = 0.50        # share of the budget for the first search
        self.LEADER_SC = 12             # owners with >= this many SCs get extra attention

    # ------------------------------------------------------------------ setup
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.static_order_streak = {p: 0 for p in game.powers if p != power_name}
        self.static_scenario = False
        self.last_enemy_targets = set()
        self.hostile_powers = set()
        self._dcache = {}
        self.home_centers = {str(c).upper() for c in game.get_centers(power_name)}
        self._stay = defaultdict(lambda: [0, 0])   # power -> [held on own SC, total]
        self._sup = defaultdict(int)               # power -> number of support orders seen
        self._last_obs = None
        self._last_plan = {}
        self._fail = defaultdict(int)              # (token, dest) -> failed attempts
        self.build_static_map(game)

    def update_game(self, all_power_orders):
        # Do not change this code.
        if getattr(self.game, 'phase_type', 'M') == 'M':
            own_centers = {str(c).upper() for c in self.game.get_centers(self.power_name)}
            own_units = set()
            for unit in self.game.get_units(self.power_name):
                parts = str(unit).split()
                if len(parts) >= 2:
                    own_units.add(parts[1].upper().split('/')[0])
            own_locations = own_centers | own_units
            self.last_enemy_targets = set()
            for p in self.static_order_streak:
                orders = all_power_orders.get(p, [])
                if not orders:
                    self.static_order_streak[p] += 1
                else:
                    self.static_order_streak[p] = 0
                for order in orders:
                    parts = str(order).split()
                    if len(parts) >= 4 and parts[2] == '-':
                        target = parts[3].upper().split('/')[0]
                        if target in own_centers:
                            self.last_enemy_targets.add(target)
                        if target in own_locations:
                            self.hostile_powers.add(p)
                    # A supported move is evidence of intent too, even if the
                    # attacking unit itself is not adjacent to our territory.
                    if ' S ' in str(order):
                        supported_move = str(order).split(' S ', 1)[1].split()
                        if len(supported_move) >= 4 and supported_move[2] == '-':
                            target = supported_move[3].upper().split('/')[0]
                            if target in own_locations:
                                self.hostile_powers.add(p)
            self.static_scenario = bool(self.static_order_streak) and all(
                streak > 0 for streak in self.static_order_streak.values()
            )
        for p in all_power_orders.keys():
            self.game.set_orders(p, all_power_orders[p])
        self.game.process()

    ###########################################################################
    # Static map
    ###########################################################################
    def build_static_map(self, game):
        locations_dict = {}
        if hasattr(game.map, "loc_type"):
            locations_dict = {str(k).upper(): str(v).upper()
                              for k, v in game.map.loc_type.items()}
        else:
            raise AttributeError("Cannot find map locations in game.map")

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

        g_army, g_fleet = nx.Graph(), nx.Graph()
        for loc_name, loc_type in locations_dict.items():
            is_sea = "SEA" in loc_type or "WATER" in loc_type
            is_coast = "COAST" in loc_type
            is_land = "LAND" in loc_type
            if is_land or is_coast:
                g_army.add_node(loc_name)
            if is_sea or is_coast:
                g_fleet.add_node(loc_name)

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
            for loc_name in locations_dict.keys():
                neighbors = set()
                if hasattr(game.map, "adjacent_provinces"):
                    for n in game.map.adjacent_provinces.get(loc_name, []):
                        neighbors.add(str(getattr(n, "name", n)).upper())
                elif hasattr(game.map, "get_neighbors"):
                    neighbors = set(str(n).upper() for n in game.map.get_neighbors(loc_name))
                for nb in neighbors:
                    if nb not in locations_dict:
                        continue
                    nb_type = locations_dict[nb]
                    if loc_name in g_army and nb in g_army:
                        if "SEA" not in nb_type and "WATER" not in nb_type:
                            g_army.add_edge(loc_name, nb)
                    if loc_name in g_fleet and nb in g_fleet:
                        if "SEA" in nb_type or "WATER" in nb_type or "COAST" in nb_type:
                            g_fleet.add_edge(loc_name, nb)

        self.graph_army, self.graph_fleet = g_army, g_fleet

        def compute_dist_table(graph, sc_list):
            sc_nodes = {sc: [n for n in graph.nodes() if n == sc or n.startswith(sc + '/')]
                        for sc in sc_list}
            table = {}
            for node in graph.nodes():
                lengths = nx.single_source_shortest_path_length(graph, node)
                table[node] = {sc: min((lengths.get(t, INF) for t in sc_nodes[sc]), default=INF)
                               for sc in sc_list}
            return table

        self.dist_army = compute_dist_table(g_army, self.supply_centers)
        self.dist_fleet = compute_dist_table(g_fleet, self.supply_centers)

        # Army distances that allow (expensive) sea hops, used only for goals so
        # that island armies still receive a target and fleets can convoy them.
        self.sea_provs = {n for n, t in locations_dict.items() if 'WATER' in t or 'SEA' in t}
        g_conv = nx.Graph()
        for u, v in g_army.edges():
            g_conv.add_edge(u, v, weight=1.0)
        for n in g_army.nodes():
            g_conv.add_node(n)
        for sea in self.sea_provs:
            if sea not in g_fleet:
                continue
            coastal = set()
            for nb in g_fleet.neighbors(sea):
                b = nb.split('/')[0]
                if b in g_army:
                    coastal.add(b)
                elif nb in g_army:
                    coastal.add(nb)
            coastal = sorted(coastal)
            for a in range(len(coastal)):
                for b in range(a + 1, len(coastal)):
                    if not g_conv.has_edge(coastal[a], coastal[b]):
                        g_conv.add_edge(coastal[a], coastal[b], weight=4.0)
        self.dist_army_conv = {n: {} for n in g_conv.nodes()}
        for sc in self.supply_centers:
            if sc not in g_conv:
                continue
            lengths = nx.single_source_dijkstra_path_length(g_conv, sc, weight='weight')
            for n in g_conv.nodes():
                self.dist_army_conv[n][sc] = lengths.get(n, INF)

        self.all_dist_army = {n: dict(l) for n, l in nx.all_pairs_shortest_path_length(g_army)}
        self.all_dist_fleet = {n: dict(l) for n, l in nx.all_pairs_shortest_path_length(g_fleet)}

        # Province-level helpers (coast suffixes removed).
        self.nodes_by_prov = defaultdict(list)
        for n in set(g_army.nodes()) | set(g_fleet.nodes()):
            self.nodes_by_prov[n.split('/')[0]].append(n)
        self.padj_fleet = defaultdict(set)
        for u, v in g_fleet.edges():
            self.padj_fleet[u.split('/')[0]].add(v.split('/')[0])
            self.padj_fleet[v.split('/')[0]].add(u.split('/')[0])
        self._dcache = {}

    ###########################################################################
    # Dynamic helpers
    ###########################################################################
    def get_dynamic_costs(self):
        friendly_occ, enemy_occ = set(), set()
        try:
            for pname, power in self.game.powers.items():
                for u in power.units:
                    parts = u.split()
                    if len(parts) < 2:
                        continue
                    loc = parts[1].upper().split('/')[0]
                    (friendly_occ if pname == self.power_name else enemy_occ).add(loc)
        except Exception:
            pass
        return friendly_occ, enemy_occ

    def get_center_owners(self):
        owners = {}
        try:
            for p in self.game.powers.keys():
                for c in self.game.get_centers(p):
                    owners[str(c).upper()] = p
        except Exception:
            pass
        return owners

    def _dist(self, kind, loc, goal, conv=False):
        """Graph distance from loc (province or coast node) to supply centre goal."""
        key = (kind, loc, goal, conv)
        v = self._dcache.get(key)
        if v is not None:
            return v
        if kind == 'F':
            table = self.dist_fleet
        else:
            table = self.dist_army_conv if conv else self.dist_army
        row = table.get(loc)
        if row is not None:
            v = row.get(goal, INF)
        else:
            v = min((table[n].get(goal, INF) for n in self.nodes_by_prov.get(loc.split('/')[0], [])
                     if n in table), default=INF)
        self._dcache[key] = v
        return v

    def _stay_rate(self, p):
        held, total = self._stay[p]
        return (held + 2.0) / (total + 3.0)

    def _observe(self, unit_tokens):
        """Learn from the previous movement phase (enemy hold rate, supports used,
        our failed moves)."""
        try:
            hist = getattr(self.game, 'order_history', None)
            if hist:
                phases = [ph for ph in hist.keys() if str(ph).endswith('M')]
                if phases and phases[-1] != self._last_obs:
                    self._last_obs = phases[-1]
                    owners = self.get_center_owners()
                    for p, orders in hist[phases[-1]].items():
                        if p == self.power_name:
                            continue
                        for o in orders:
                            parts = str(o).split()
                            if len(parts) < 3:
                                continue
                            if parts[2] == 'S':
                                self._sup[p] += 1
                            prov = parts[1].upper().split('/')[0]
                            if prov in self.supply_centers and owners.get(prov) == p:
                                rec = self._stay[p]
                                rec[1] += 1
                                if parts[2] != '-':
                                    rec[0] += 1
                    # our own bounced moves
                    for tok, order in self._last_plan.items():
                        parts = order.split()
                        if len(parts) >= 4 and parts[2] == '-' and 'VIA' not in parts:
                            key = (tok, parts[3].upper().split('/')[0])
                            if tok in unit_tokens:
                                self._fail[key] += 1
                            else:
                                self._fail.pop(key, None)
            for key in list(self._fail):
                if key[0] not in unit_tokens:
                    del self._fail[key]
        except Exception:
            pass

    def _smart_powers(self):
        """Opponents that look competent: they use supports, or have attacked us."""
        smart = set()
        for p in self.static_order_streak:
            if self.static_order_streak.get(p, 0) > 0:
                continue                      # sends no orders at all -> passive
            if self._sup.get(p, 0) >= 1 or p in self.hostile_powers:
                smart.add(p)
        return smart

    ###########################################################################
    # Internal adjudicator
    ###########################################################################
    def _convoy_ok(self, i, dest, prov, orders, dead):
        fleets = [j for j in range(len(orders))
                  if orders[j][0] == 'C' and orders[j][1] == i and orders[j][2] == dest
                  and j not in dead]
        if not fleets:
            return False
        start_adj = self.padj_fleet.get(prov[i], ())
        reach = [j for j in fleets if prov[j] in start_adj]
        seen = set(reach)
        while reach:
            j = reach.pop()
            if dest in self.padj_fleet.get(prov[j], ()):
                return True
            for k in fleets:
                if k not in seen and prov[k] in self.padj_fleet.get(prov[j], ()):
                    seen.add(k)
                    reach.append(k)
        return False

    def _adjudicate(self, prov, pw, orders):
        """
        Simplified Diplomacy adjudication.
        orders: ('H',), ('M', dest, via), ('SH', t), ('SM', t, dest), ('C', army, dest)
        Returns (final_province list, dislodged list).
        """
        n = len(orders)
        occ = {prov[i]: i for i in range(n)}
        dead = set()
        has_via = any(o[0] == 'M' and o[2] for o in orders)
        for attempt in range(2):
            moving = [None] * n
            for i in range(n):
                o = orders[i]
                if o[0] == 'M':
                    if not o[2] or self._convoy_ok(i, o[1], prov, orders, dead):
                        moving[i] = o[1]
            atk = {}
            for i in range(n):
                if moving[i] is not None:
                    atk.setdefault(moving[i], []).append(i)
            ms = [1] * n
            hs = [1] * n
            for i in range(n):
                o = orders[i]
                k = o[0]
                if k == 'SM':
                    t = o[1]
                    if moving[t] != o[2]:
                        continue
                elif k == 'SH':
                    t = o[1]
                    if moving[t] is not None:
                        continue
                else:
                    continue
                cut = False
                for j in atk.get(prov[i], ()):
                    if pw[j] != pw[i] and not (k == 'SM' and prov[j] == o[2]):
                        cut = True
                        break
                if cut:
                    continue
                if k == 'SM':
                    ms[t] += 1
                else:
                    hs[t] += 1
            movers = [i for i in range(n) if moving[i] is not None]
            succ = {i: True for i in movers}
            for _ in range(4):
                new = {}
                for i in movers:
                    d = moving[i]
                    s = ms[i]
                    ok = True
                    for j in atk[d]:
                        if j != i and ms[j] >= s:
                            ok = False
                            break
                    if ok:
                        u = occ.get(d)
                        if u is not None and u != i:
                            mu = moving[u]
                            if mu is not None:
                                if mu == prov[i] and not (orders[i][2] or orders[u][2]):
                                    resist = ms[u]
                                elif succ.get(u, True):
                                    resist = 0
                                else:
                                    resist = 1
                            else:
                                resist = hs[u]
                            if s <= resist:
                                ok = False
                            elif resist > 0 and pw[u] == pw[i]:
                                ok = False
                    new[i] = ok
                if new == succ:
                    break
                succ = new
            disl = [False] * n
            for i in movers:
                if succ[i]:
                    u = occ.get(moving[i])
                    if u is not None and u != i and not (moving[u] is not None and succ.get(u)):
                        disl[u] = True
            if has_via and attempt == 0:
                newdead = {u for u in range(n) if disl[u] and orders[u][0] == 'C'}
                if newdead - dead:
                    dead |= newdead
                    continue
            break
        fpos = [moving[i] if (moving[i] is not None and succ[i]) else prov[i] for i in range(n)]
        return fpos, disl

    ###########################################################################
    # Goal assignment
    ###########################################################################
    def _assign_goals(self, infos, owners, enemy_occ):
        me = self.power_name
        m = len(infos)
        targets = [sc for sc in self.supply_centers if owners.get(sc) != me]
        goals = [None] * m
        claimed = set()
        # Units standing on a centre we do not own yet keep it (capture pending).
        for i, (kind, loc, prov) in enumerate(infos):
            if prov in targets and prov not in claimed:
                goals[i] = prov
                claimed.add(prov)
        pairs = []
        cost = {}
        for i, (kind, loc, prov) in enumerate(infos):
            if goals[i] is not None:
                continue
            for sc in targets:
                if sc in claimed:
                    continue
                d = self._dist(kind, loc, sc, True)
                if d == INF:
                    continue
                c = d - (2.0 if owners.get(sc) is None else 0.0)
                if sc in enemy_occ:
                    c += 1.5
                pairs.append((c, d, i, sc))
                cost[(i, sc)] = c
        pairs.sort()
        assign = {}
        used = set()
        for c, d, i, sc in pairs:
            if i not in assign and sc not in used:
                assign[i] = sc
                used.add(sc)
        keys = list(assign)
        for _ in range(3):
            changed = False
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    x, y = keys[a], keys[b]
                    gx, gy = assign[x], assign[y]
                    cur = cost.get((x, gx), INF) + cost.get((y, gy), INF)
                    sw = cost.get((x, gy), INF) + cost.get((y, gx), INF)
                    if sw + 1e-9 < cur:
                        assign[x], assign[y] = gy, gx
                        changed = True
            if not changed:
                break
        for i, sc in assign.items():
            goals[i] = sc
        # Leftover units head for the cheapest centre (may duplicate a goal so
        # that several units converge on a defended centre).
        for i, (kind, loc, prov) in enumerate(infos):
            if goals[i] is not None:
                continue
            best, bc = None, INF
            for sc in targets:
                c = cost.get((i, sc))
                if c is None:
                    d = self._dist(kind, loc, sc, True)
                    if d == INF:
                        continue
                    c = d - (2.0 if owners.get(sc) is None else 0.0)
                if c < bc:
                    best, bc = sc, c
            goals[i] = best
        return goals

    ###########################################################################
    # Enemy behaviour model
    ###########################################################################
    def _enemy_scenarios(self, enemies, m, owners, my_prov_set):
        """
        enemies: list of (power, kind, loc, prov). Returns three order lists
        (hold, greedy advance, coordinated attack) indexed like enemies.
        """
        me = self.power_name
        n = len(enemies)
        hold = [('H',)] * n
        greedy = [('H',)] * n
        attack = [('H',)] * n
        power_targets = {}
        by_power_prov = defaultdict(set)
        for (p, k, loc, prov) in enemies:
            by_power_prov[p].add(prov)
        passive = {p for p, st in self.static_order_streak.items() if st > 0}
        for idx, (p, k, loc, prov) in enumerate(enemies):
            graph = self.graph_fleet if k == 'F' else self.graph_army
            if loc not in graph or p in passive:
                continue
            nbrs = sorted({x.split('/')[0] for x in graph.neighbors(loc)})
            # greedy advance
            if p not in power_targets:
                power_targets[p] = [sc for sc in self.supply_centers if owners.get(sc) != p]
            tg = power_targets[p]
            if prov in tg or (owners.get(prov) == p and self._stay_rate(p) >= 0.5):
                pass  # standing on a centre it wants / defends: hold
            else:
                best, bd = None, INF
                cur_d = min((self._dist(k, loc, sc) for sc in tg), default=INF)
                for q in nbrs:
                    if q in by_power_prov[p]:
                        continue
                    d = min((self._dist(k, q, sc) for sc in tg), default=INF)
                    if d < bd:
                        best, bd = q, d
                if best is not None and bd < cur_d:
                    greedy[idx] = ('M', best, False)
            # coordinated attack on us / neutral centres
            best, bs = None, 0.0
            for q in nbrs:
                if q in by_power_prov[p]:
                    continue
                s = 0.0
                if owners.get(q) == me:
                    s += 8.0
                if q in my_prov_set:
                    s += 4.0
                if q in self.supply_centers and owners.get(q) is None:
                    s += 3.0
                if s > bs:
                    best, bs = q, s
            if best is not None:
                attack[idx] = ('M', best, False)
        # convert duplicate attackers of the same power into supporters
        seen = {}
        for idx, o in enumerate(attack):
            if o[0] != 'M':
                continue
            key = (enemies[idx][0], o[1])
            if key in seen:
                attack[idx] = ('SM', m + seen[key], o[1])
            else:
                seen[key] = idx
        return hold, greedy, attack

    ###########################################################################
    # Movement planner
    ###########################################################################
    def plan_movement(self, unit_options, deadline):
        game = self.game
        me = self.power_name
        phase = game.get_current_phase()
        spring = phase.startswith('S')
        owners = self.get_center_owners()
        my_sc = {sc for sc, p in owners.items() if p == me}
        counts = defaultdict(int)
        for p in owners.values():
            counts[p] += 1

        tokens = list(unit_options.keys())
        m = len(tokens)
        infos = []
        for t in tokens:
            kind, loc = t.split()[:2]
            infos.append((kind, loc.upper(), loc.upper().split('/')[0]))
        tok_index = {t: i for i, t in enumerate(tokens)}
        pk_index = {(infos[i][0], infos[i][2]): i for i in range(m)}

        def find_unit(kind, loc):
            i = tok_index.get(kind + ' ' + loc)
            if i is None:
                i = pk_index.get((kind, loc.split('/')[0]))
            return i

        # enemy units
        enemies = []
        pids = {me: 0}
        for pname, power in game.powers.items():
            if pname == me:
                continue
            pids.setdefault(pname, len(pids))
            for u in power.units:
                parts = str(u).split()
                if len(parts) < 2 or parts[0].startswith('*'):
                    continue
                enemies.append((pname, parts[0], parts[1].upper(), parts[1].upper().split('/')[0]))
        enemy_occ = {e[3] for e in enemies}
        my_prov_set = {p for (_, _, p) in infos}

        # ---- candidate orders
        cands = [[] for _ in range(m)]
        move_lookup = {}      # (i, dest) -> (tuple, str)
        supports = []         # (s, t, dest, tuple, str)
        via_moves = []        # (i, dest, tuple, str)
        conv_orders = []      # (fleet i, army j, dest, tuple, str)
        zone = set(my_prov_set) | set(my_sc)
        goals = self._assign_goals(infos, owners, enemy_occ)
        for i, t in enumerate(tokens):
            kind, loc, prov = infos[i]
            goal = goals[i]
            seen = {}
            for s in unit_options[t]:
                p = s.split()
                if len(p) == 3 and p[2] == 'H':
                    seen[('H',)] = (('H',), s)
                elif len(p) >= 4 and p[2] == '-':
                    dest = p[3].split('/')[0]
                    zone.add(dest)
                    if 'VIA' in p:
                        via_moves.append((i, dest, ('M', dest, True), s))
                        continue
                    tup = ('M', dest, False)
                    if tup in seen and goal is not None:
                        old = seen[tup][1].split()[3]
                        if self._dist(kind, p[3].upper(), goal, True) < \
                                self._dist(kind, old.upper(), goal, True):
                            seen[tup] = (tup, s)
                    else:
                        seen.setdefault(tup, (tup, s))
                elif len(p) >= 5 and p[2] == 'S':
                    j = find_unit(p[3], p[4].upper())
                    if j is None or j == i:
                        continue
                    if len(p) == 5:
                        supports.append((i, j, None, ('SH', j), s))
                    elif len(p) == 7 and p[5] == '-':
                        dest = p[6].split('/')[0]
                        supports.append((i, j, dest, ('SM', j, dest), s))
                elif len(p) >= 7 and p[2] == 'C':
                    j = find_unit(p[3], p[4].upper())
                    if j is None:
                        continue
                    dest = p[6].split('/')[0]
                    conv_orders.append((i, j, dest, ('C', j, dest), s))
            cands[i] = list(seen.values())
            for tup, s in cands[i]:
                if tup[0] == 'M':
                    move_lookup[(i, tup[1])] = (tup, s)

        interesting = ({sc for sc in self.supply_centers if owners.get(sc) != me}
                       | my_sc | enemy_occ)
        macros = []
        for (s, t, dest, tup, sstr) in supports:
            if dest is None:
                # support-hold only for units standing on a centre
                if infos[t][2] in self.supply_centers:
                    cands[s].append((tup, sstr))
            elif dest in interesting:
                mv = move_lookup.get((t, dest))
                if mv is not None:
                    macros.append([(t, mv[0], mv[1]), (s, tup, sstr)])
        for (a, dest, tup, s) in via_moves:
            if dest not in interesting:
                continue
            fl = [c for c in conv_orders if c[1] == a and c[2] == dest]
            for (f, j, d, ctup, cs) in fl:
                macros.append([(a, tup, s), (f, ctup, cs)])
            if len(fl) > 1:
                macros.append([(a, tup, s)] + [(f, ctup, cs) for (f, j, d, ctup, cs) in fl])

        # ---- enemy relevance & scenarios
        rel = []
        for e in enemies:
            p, k, loc, prov = e
            graph = self.graph_fleet if k == 'F' else self.graph_army
            if prov in zone:
                rel.append(e)
                continue
            if loc in graph and any(x.split('/')[0] in zone for x in graph.neighbors(loc)):
                rel.append(e)
        if self.opponent_model:
            e_hold, e_greedy, e_attack = self._enemy_scenarios(rel, m, owners, my_prov_set)
            scen_orders = [e_hold, e_greedy, e_attack]
            weights = list(self.SCENARIO_WEIGHTS)      # mutated in place later
        else:
            # Technique 2 only: single enemy model, every relevant unit holds.
            e_hold = [('H',)] * len(rel)
            e_attack = e_hold
            scen_orders = [e_hold]
            weights = [1.0]
        prov_all = [infos[i][2] for i in range(m)] + [e[3] for e in rel]
        pw_all = [0] * m + [pids[e[0]] for e in rel]

        # ---- evaluation
        capval = {}
        for sc in self.supply_centers:
            o = owners.get(sc)
            if o == me:
                continue
            v = self.SPRING_CAPTURE if spring else self.FALL_CAPTURE
            if o is None:
                v += 1.0
            else:
                v += 1.5 + min(4.0, max(0, counts[o] - 5) * 0.5)
                if self.opponent_model and o in self.hostile_powers:
                    v += 1.0
                if counts[o] >= self.LEADER_SC:
                    # containing a runaway leader matters more than a normal grab
                    v += 1.5 + min(4.0, 0.75 * (counts[o] - self.LEADER_SC))
            capval[sc] = v
        sc_loss = self.SC_LOSS_SPRING if spring else self.SC_LOSS_FALL
        n_my_sc = len(my_sc)
        kinds = [infos[i][0] for i in range(m)]
        enemy_sc_units = [e[3] in self.supply_centers for e in rel]
        rel_n = len(rel)
        dist = self._dist
        progress = self.PROGRESS

        conv_armies = []
        for i in range(m):
            if kinds[i] == 'A' and goals[i] is not None and \
                    self._dist('A', infos[i][1], goals[i]) == INF:
                conv_armies.append(i)
        fleet_units = [i for i in range(m) if kinds[i] == 'F']
        sea_provs = self.sea_provs
        padj = self.padj_fleet

        fail_pen = {}
        for i, t in (enumerate(tokens) if self.opponent_model else ()):
            for (tk, dest), cnt in self._fail.items():
                if tk == t:
                    fail_pen[(i, dest)] = 2.5 * min(cnt, 3)

        def raw_score(my_orders):
            total = 0.0
            pen = 0.0
            if fail_pen:
                sup_t = {o[1] for o in my_orders if o[0] == 'SM'}
                for i in range(m):
                    o = my_orders[i]
                    if o[0] == 'M' and i not in sup_t:
                        pen += fail_pen.get((i, o[1]), 0.0)
            for w, eo in zip(weights, scen_orders):
                orders = my_orders + eo
                fpos, disl = self._adjudicate(prov_all, pw_all, orders)
                v = 0.0
                captured = 0
                dis_mine = 0
                for i in range(m):
                    if disl[i]:
                        v -= self.UNIT_LOSS
                        dis_mine += 1
                        continue
                    f = fpos[i]
                    g = goals[i]
                    if g is not None:
                        d = dist(kinds[i], f, g, True)
                        v -= progress * (d if d < INF else 12)
                    cv = capval.get(f)
                    if cv is not None:
                        v += cv
                        captured += 1
                if conv_armies:
                    for i in conv_armies:
                        if disl[i]:
                            continue
                        adj = padj.get(fpos[i], ())
                        for j in fleet_units:
                            if not disl[j] and fpos[j] in adj and fpos[j] in sea_provs:
                                v += 2.0
                                break
                for k in range(rel_n):
                    if disl[m + k]:
                        v += self.DISLODGE_BONUS + (2.0 if enemy_sc_units[k] else 0.0)
                occ = {}
                for i in range(m + rel_n):
                    if not disl[i]:
                        occ[fpos[i]] = i
                lost = 0
                vacant_home = 0
                for sc in my_sc:
                    u = occ.get(sc)
                    if u is not None and u >= m:
                        v -= sc_loss
                        lost += 1
                    elif u is None and sc in self.home_centers:
                        vacant_home += 1
                if not spring:
                    room = (n_my_sc + captured - lost) - (m - dis_mine)
                    if room > 0:
                        v += self.BUILD_ROOM_BONUS * min(room, vacant_home)
                total += w * v
            return total - pen

        memo = {}

        def score(my_orders):
            key = tuple(my_orders)
            r = memo.get(key)
            if r is None:
                r = raw_score(my_orders)
                memo[key] = r
            return r

        # ---- local search
        base = [c[0] for c in cands]
        for i in range(m):
            for tup, st in cands[i]:
                if tup == ('H',):
                    base[i] = (tup, st)
                    break
        order_units = sorted(range(m), key=lambda i: (
            goals[i] is None,
            dist(infos[i][0], infos[i][1], goals[i], True) if goals[i] else 0))

        def descend(cur, cur_str, best, units_order, limit):
            for _ in range(self.MAX_PASSES):
                changed = False
                for i in units_order:
                    if time.perf_counter() > limit:
                        return best
                    bi, bv = None, best
                    for tup, st in cands[i]:
                        if tup == cur[i]:
                            continue
                        old = cur[i]
                        cur[i] = tup
                        v = score(cur)
                        cur[i] = old
                        if v > bv + 1e-6:
                            bv, bi = v, (tup, st)
                    if bi is not None:
                        cur[i], cur_str[i] = bi
                        best = bv
                        changed = True
                for macro in macros:
                    if time.perf_counter() > limit:
                        return best
                    old = [(i, cur[i], cur_str[i]) for i, _, _ in macro]
                    for i, tup, st in macro:
                        cur[i], cur_str[i] = tup, st
                    v = score(cur)
                    if v > best + 1e-6:
                        best = v
                        changed = True
                    else:
                        for i, tup, st in old:
                            cur[i], cur_str[i] = tup, st
                if not changed:
                    break
            return best

        rng = random.Random(len(phase) * 7919 + m)

        def search(start_plan, start_str, limit, max_restarts):
            cur, cur_str = list(start_plan), list(start_str)
            best = descend(cur, cur_str, score(cur), order_units, limit)
            best_plan, best_str = list(cur), list(cur_str)
            restarts = 0
            while restarts < max_restarts and time.perf_counter() < limit:
                restarts += 1
                cur, cur_str = list(best_plan), list(best_str)
                k = max(1, m // 4)
                for i in rng.sample(range(m), min(k, m)):
                    tup, st = rng.choice(cands[i])
                    cur[i], cur_str[i] = tup, st
                units_order = list(range(m))
                rng.shuffle(units_order)
                v = descend(cur, cur_str, score(cur), units_order, limit)
                if v > best + 1e-6:
                    best, best_plan, best_str = v, list(cur), list(cur_str)
            return best, best_plan, best_str

        # ---- opponent best-response machinery ---------------------------------
        smart = self._smart_powers()
        br_powers = sorted({e[0] for e in rel if e[0] in smart})
        use_br = (self.opponent_model and self.BR_WEIGHT > 0 and rel_n > 0
                  and bool(br_powers))

        t_begin = time.perf_counter()
        total_time = max(0.0, deadline - t_begin)
        phase1_limit = deadline if not use_br else t_begin + total_time * self.PHASE1_SHARE

        start_plan = [b[0] for b in base]
        start_str = [b[1] for b in base]
        best, best_plan, best_str = search(start_plan, start_str, phase1_limit, self.MAX_RESTARTS)

        if use_br and time.perf_counter() < deadline - 0.03:
            try:
                br_limit = min(deadline, time.perf_counter() + total_time * self.BR_TIME_SHARE)
                br_orders = self._enemy_best_response(
                    best_plan, rel, br_powers, e_attack, m, owners, my_sc, my_prov_set,
                    prov_all, pw_all, infos, spring, br_limit)
                if br_orders is not None:
                    w = self.BR_WEIGHT
                    weights[:] = [x * (1.0 - w) for x in self.SCENARIO_WEIGHTS] + [w]
                    scen_orders.append(br_orders)
                    memo.clear()
                    b2, p2, s2 = search(best_plan, best_str, deadline, self.MAX_RESTARTS)
                    best_plan, best_str = p2, s2
            except Exception:
                pass
        return best_str

    ###########################################################################
    # Enemy best response (fictitious-play step)
    ###########################################################################
    def _enemy_best_response(self, my_plan, rel, powers, base_eo, m, owners, my_sc,
                             my_prov_set, prov_all, pw_all, infos, spring, limit):
        """
        For every listed opponent power, improve its orders by coordinate descent
        (single-unit changes + move/support pairs) against OUR current plan.
        Returns an order list aligned with `rel` (enemy index k -> global m+k).
        """
        me = self.power_name
        rel_n = len(rel)
        sc_set = set(self.supply_centers)
        ecap = 0.7 * (self.SPRING_CAPTURE if spring else self.FALL_CAPTURE)
        interest = sc_set | my_prov_set | set(my_sc)

        occ_by_power = defaultdict(set)
        for (p, kd, loc, prov) in rel:
            occ_by_power[p].add(prov)

        nbrs = []
        for (p, kd, loc, prov) in rel:
            graph = self.graph_fleet if kd == 'F' else self.graph_army
            if loc in graph:
                nbrs.append(sorted({x.split('/')[0] for x in graph.neighbors(loc)}))
            else:
                nbrs.append([])

        ecands = []
        for k in range(rel_n):
            p = rel[k][0]
            out = [('H',)]
            for q in nbrs[k]:
                if q in occ_by_power[p]:
                    continue
                out.append(('M', q, False))
            for k2 in range(rel_n):
                if k2 == k or rel[k2][0] != p:
                    continue
                if rel[k2][3] in nbrs[k] and rel[k2][3] in interest:
                    out.append(('SH', m + k2))
                for q in nbrs[k2]:
                    if q in nbrs[k] and q in interest and q not in occ_by_power[p]:
                        out.append(('SM', m + k2, q))
            ecands.append(out)

        emacros = defaultdict(list)
        for k2 in range(rel_n):
            p = rel[k2][0]
            for q in nbrs[k2]:
                if q not in interest or q in occ_by_power[p]:
                    continue
                for k in range(rel_n):
                    if k != k2 and rel[k][0] == p and q in nbrs[k]:
                        emacros[p].append((k2, ('M', q, False), k, ('SM', m + k2, q)))

        ptargets = {}
        near_memo = {}
        dist = self._dist

        def enemy_near(p, kd, q):
            key = (p, kd, q)
            v = near_memo.get(key)
            if v is None:
                tg = ptargets.get(p)
                if tg is None:
                    tg = [sc for sc in self.supply_centers if owners.get(sc) != p]
                    ptargets[p] = tg
                v = min((dist(kd, q, sc) for sc in tg), default=INF)
                near_memo[key] = v
            return v

        def enemy_value(idxs, p, eo):
            fpos, disl = self._adjudicate(prov_all, pw_all, my_plan + eo)
            v = 0.0
            for k in idxs:
                i = m + k
                if disl[i]:
                    v -= self.UNIT_LOSS
                    continue
                f = fpos[i]
                if f in sc_set and owners.get(f) != p:
                    c = ecap + (0.5 if owners.get(f) is None else 0.0)
                    if owners.get(f) == me:
                        c += 4.0
                    v += c
                d = enemy_near(p, rel[k][1], f)
                v -= 0.5 * (d if d < INF else 8)
            for i in range(m):
                if disl[i]:
                    v += 3.0 + (2.0 if infos[i][2] in my_sc else 0.0)
            return v

        eo = list(base_eo)
        for p in powers:
            idxs = [k for k in range(rel_n) if rel[k][0] == p]
            if not idxs:
                continue
            best = enemy_value(idxs, p, eo)
            for _ in range(2):
                changed = False
                for k in idxs:
                    if time.perf_counter() > limit:
                        return eo
                    bo = None
                    for c in ecands[k]:
                        if c == eo[k]:
                            continue
                        old = eo[k]
                        eo[k] = c
                        v = enemy_value(idxs, p, eo)
                        eo[k] = old
                        if v > best + 1e-6:
                            best, bo = v, c
                    if bo is not None:
                        eo[k] = bo
                        changed = True
                for (k2, mv, k, sup) in emacros[p]:
                    if time.perf_counter() > limit:
                        return eo
                    o2, o1 = eo[k2], eo[k]
                    eo[k2], eo[k] = mv, sup
                    v = enemy_value(idxs, p, eo)
                    if v > best + 1e-6:
                        best = v
                        changed = True
                    else:
                        eo[k2], eo[k] = o2, o1
                if not changed:
                    break
        return eo

    ###########################################################################
    # Non-movement phases
    ###########################################################################
    def handle_non_movement(self):
        phase_type = getattr(self.game, 'phase_type', 'M')
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = []

        center_owners = self.get_center_owners()
        if phase_type == 'R':
            friendly_occ, enemy_occ = self.get_dynamic_costs()
            my_centers = {c for c, p in center_owners.items() if p == self.power_name}
        else:
            friendly_occ, enemy_occ = set(), set()
            my_centers = set()
        unowned_centers = [c for c in self.supply_centers
                           if center_owners.get(c) != self.power_name]
        unowned_set = set(unowned_centers)
        try:
            fleet_count = sum(str(u).upper().startswith('F ')
                              for u in self.game.get_units(self.power_name))
        except Exception:
            fleet_count = 0

        def strs(loc):
            return [o.as_string() if hasattr(o, 'as_string') else str(o)
                    for o in all_possible_orders.get(loc, [])]

        if phase_type == 'A':
            build_count = len(self.game.get_centers(self.power_name)) - len(
                self.game.get_units(self.power_name))
            if build_count > 0:
                candidates = []
                for loc in orderable_locations:
                    for order in strs(loc):
                        parts = order.split()
                        if not parts or parts[-1] != 'B':
                            continue
                        table = self.dist_fleet if parts[0] == 'F' else self.dist_army
                        distances = table.get(parts[1].upper(), {})
                        near = [distances.get(c, INF) for c in unowned_centers]
                        near = [d for d in near if d < INF]
                        candidates.append((parts[1].upper().split('/')[0], order,
                                           min(near, default=99),
                                           -sum(1 for d in near if d <= 2)))
                builds = []
                remaining = list(candidates)
                ratio = 0.45 if self.power_name == 'ENGLAND' else 0.3
                unit_count = len(self.game.get_units(self.power_name))
                while len(builds) < build_count and remaining:
                    want = 'F' if (fleet_count / max(1, unit_count + 1)) < ratio else 'A'
                    pool_ = [it for it in remaining if it[1].startswith(want + ' ')] or remaining
                    selected = min(pool_, key=lambda it: (it[2], it[3], it[1]))
                    builds.append(selected[1])
                    unit_count += 1
                    if selected[1].startswith('F '):
                        fleet_count += 1
                    remaining = [it for it in remaining if it[0] != selected[0]]
                return builds
            if build_count < 0:
                candidates = []
                for loc in orderable_locations:
                    for order in [s for s in strs(loc) if s.split() and s.split()[-1] == 'D']:
                        parts = order.split()
                        table = self.dist_fleet if parts[0] == 'F' else self.dist_army
                        distances = table.get(parts[1].upper(), {})
                        fd = min((distances.get(c, 99) for c in unowned_centers), default=99)
                        on_center = (center_owners.get(parts[1].upper().split('/')[0]) ==
                                     self.power_name)
                        candidates.append((fd - (1 if on_center else 0), order))
                candidates.sort(key=lambda it: (it[0], it[1]), reverse=True)
                return [o for _, o in candidates[:-build_count]]
            return []

        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue
            strings = sorted(strs(loc))
            if phase_type == 'R':
                retreats = [s for s in strings if ' R ' in s]
                if retreats:
                    best_retreat, best_score = None, INF
                    for s in retreats:
                        parts = s.split()
                        if len(parts) < 4:
                            continue
                        dest = parts[3].upper()
                        province = dest.split('/')[0]
                        table = self.dist_fleet if parts[0] == "F" else self.dist_army
                        score = 0.0
                        if province in enemy_occ:
                            score += 1000.0
                        if province in friendly_occ:
                            score += 500.0
                        if my_centers:
                            dists = [table.get(dest, table.get(province, {})).get(c, INF)
                                     for c in my_centers]
                            bd = min(dists)
                            if bd != INF:
                                score += bd
                        # retreating onto a free centre we do not own is a free grab
                        if province in unowned_set and province not in enemy_occ \
                                and province not in friendly_occ:
                            score -= 1.5
                        if score < best_score:
                            best_score, best_retreat = score, s
                    power_orders.append(best_retreat if best_retreat else retreats[0])
                else:
                    disbands = [s for s in strings if s.split() and s.split()[-1] == 'D']
                    power_orders.append(disbands[0] if disbands else strings[0])
            else:
                power_orders.append(strings[0])
        return power_orders

    ###########################################################################
    # Entry point
    ###########################################################################
    def get_actions(self):
        t0 = time.perf_counter()
        if getattr(self.game, "phase_type", "M") != "M":
            return self.handle_non_movement()
        try:
            all_possible_orders = self.game.get_all_possible_orders()
            orderable_locations = self.game.get_orderable_locations(self.power_name)
        except Exception:
            return []
        unit_options = defaultdict(list)
        for loc in orderable_locations:
            for order in all_possible_orders.get(loc, []):
                s = order.as_string() if hasattr(order, "as_string") else str(order)
                parts = s.split()
                if len(parts) >= 2:
                    unit_options[parts[0] + " " + parts[1]].append(s)
        for opts in unit_options.values():
            opts.sort()
        if not unit_options:
            return []

        if self.opponent_model:
            self._observe(set(unit_options.keys()))
        book = self._opening_book().get(self.power_name, {})
        if self.game.get_current_phase() == 'S1901M':
            orders = []
            for token, options in unit_options.items():
                d = book.get(token)
                if d and d in options:
                    orders.append(d)
                else:
                    orders = None
                    break
            if orders:
                self._last_plan = dict(zip(unit_options.keys(), orders))
                return orders
        try:
            deadline = t0 + self.SEARCH_BUDGET
            result = list(self.plan_movement(unit_options, deadline))
            self._last_plan = dict(zip(unit_options.keys(), result))
            return result
        except Exception:
            import traceback
            traceback.print_exc()
            return [next((o for o in opts if o.split()[-1] == 'H'), opts[0])
                    for opts in unit_options.values()]

    @staticmethod
    def _opening_book():
        return {
            'AUSTRIA': {'A VIE': 'A VIE - GAL', 'A BUD': 'A BUD - SER', 'F TRI': 'F TRI - ALB'},
            'ENGLAND': {'F EDI': 'F EDI - NTH', 'F LON': 'F LON - ENG', 'A LVP': 'A LVP - WAL'},
            'FRANCE': {'F BRE': 'F BRE - MAO', 'A MAR': 'A MAR - SPA', 'A PAR': 'A PAR - PIC'},
            'GERMANY': {'A BER': 'A BER - KIE', 'F KIE': 'F KIE - DEN', 'A MUN': 'A MUN - RUH'},
            'ITALY': {'F NAP': 'F NAP - ION', 'A ROM': 'A ROM - APU', 'A VEN': 'A VEN - TYR'},
            'RUSSIA': {'A MOS': 'A MOS - STP', 'F SEV': 'F SEV - RUM',
                       'F STP/SC': 'F STP/SC - BOT', 'A WAR': 'A WAR - GAL'},
            'TURKEY': {'F ANK': 'F ANK - BLA', 'A CON': 'A CON - BUL', 'A SMY': 'A SMY - ARM'},
        }


###############################################################################
# Technique 2 (not called by the game): joint-order planning without the
# Technique 3 opponent model.
###############################################################################
class Technique2Agent(StudentAgent):
    def __init__(self, agent_name="Technique2Agent"):
        super().__init__(agent_name, opponent_model=False)


###############################################################################
# Basic technique (not called by the game): per-unit A* toward the nearest
# supply centre, executing the first step of the path and replanning each phase.
###############################################################################
class BasicAStarAgent(Agent):
    """
    A* path planning accounting for certain risk.

    1. Build army/fleet graphs and precompute distances to all supply centres.
    2. For every unit, pick the nearest supply centres as targets.
    3. Run A* from the unit to each target; the heuristic is the static graph
       distance, and enemy-occupied provinces are penalised or avoided.
    4. Execute the first step of the best path found, then replan next phase.
    """

    MAX_TARGETS = 5
    BLOCK_ENEMY_TRANSIT = True   # basic A* never routes through enemy units

    def __init__(self, agent_name="BasicAStarAgent"):
        super().__init__(agent_name)
        self.game = None
        self.power_name = None
        self.graph_army = None
        self.graph_fleet = None
        self.dist_army = {}
        self.dist_fleet = {}
        self.supply_centers = []
        self.TIME_BUDGET = 0.85
        self.MAX_NODES_PER_UNIT = 300
        self.ENEMY_PENALTY_PASS = 15.0
        self.ENEMY_PENALTY_ATTACK = 5.0
        self.MAX_TARGETS_PER_UNIT = self.MAX_TARGETS

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_static_map(game)

    def update_game(self, all_power_orders):
        # Do not change this code.
        for p in all_power_orders.keys():
            self.game.set_orders(p, all_power_orders[p])
        self.game.process()

    # ------------------------------------------------------------ static map
    def _collect_supply_centers(self, game, locations_dict):
        names = []
        for sc in getattr(game.map, 'scs', []):
            name = str(getattr(sc, 'name', sc)).upper()
            if name in locations_dict:
                names.append(name)
        return names

    def build_static_map(self, game):
        locations_dict = {str(k).upper(): str(v).upper()
                          for k, v in game.map.loc_type.items()}
        self.supply_centers = self._collect_supply_centers(game, locations_dict)

        g_army, g_fleet = nx.Graph(), nx.Graph()
        for loc, ltype in locations_dict.items():
            is_sea = 'SEA' in ltype or 'WATER' in ltype
            is_coast = 'COAST' in ltype
            is_land = 'LAND' in ltype
            if is_land or is_coast:
                g_army.add_node(loc)
            if is_sea or is_coast:
                g_fleet.add_node(loc)

        locs = list(locations_dict.keys())
        for i in locs:
            for j in locs:
                if i == j:
                    continue
                try:
                    if i in g_army and j in g_army and game.map.abuts('A', i, '-', j):
                        g_army.add_edge(i, j)
                    if i in g_fleet and j in g_fleet and game.map.abuts('F', i, '-', j):
                        g_fleet.add_edge(i, j)
                except Exception:
                    pass
        self.graph_army, self.graph_fleet = g_army, g_fleet

        # Heuristic table: BFS shortest-path length from every node to every SC.
        def compute_dist_table(graph, sc_list):
            table = {}
            for node in graph.nodes():
                lengths = nx.single_source_shortest_path_length(graph, node)
                table[node] = {sc: lengths.get(sc, float('inf')) for sc in sc_list}
            return table

        self.dist_army = compute_dist_table(g_army, self.supply_centers)
        self.dist_fleet = compute_dist_table(g_fleet, self.supply_centers)

    # -------------------------------------------------------- dynamic helpers
    def get_dynamic_costs(self):
        friendly_occ, enemy_occ = set(), set()
        try:
            for pname, power in self.game.powers.items():
                for u in power.units:
                    parts = str(u).split()
                    if len(parts) < 2:
                        continue
                    loc = parts[1].upper()
                    (friendly_occ if pname == self.power_name else enemy_occ).add(loc)
        except Exception:
            pass
        return friendly_occ, enemy_occ

    def get_center_owners(self):
        owners = {}
        try:
            for p in self.game.powers.keys():
                for c in self.game.get_centers(p):
                    owners[str(c).upper()] = p
        except Exception:
            pass
        return owners

    # ------------------------------------------------------ basic targeting
    def choose_targets(self, loc, kind, friendly_occ, enemy_occ, reserved_dests,
                       center_owners):
        """Basic targeting: nearest supply centres by static graph distance only.
        center_owners is deliberately unused (ownership is Technique 1)."""
        dist_table = self.dist_army if kind == 'Army' else self.dist_fleet
        if loc not in dist_table:
            return []
        scored = []
        for sc in self.supply_centers:
            if sc in reserved_dests or sc in friendly_occ:
                continue
            d = dist_table[loc].get(sc, float('inf'))
            if d == float('inf'):
                continue
            scored.append((d, sc))
        scored.sort(key=lambda x: x[0])
        return [sc for _, sc in scored[:self.MAX_TARGETS_PER_UNIT]]

    # ----------------------------------------------------------------- A*
    def astar_search(self, start_loc, goal_loc, unit_kind, friendly_occ, enemy_occ,
                     deadline):
        """A* from start_loc to goal_loc. g = step costs plus enemy penalties,
        h = precomputed static distance (admissible: it ignores all units)."""
        start_loc, goal_loc = str(start_loc).upper(), str(goal_loc).upper()
        if start_loc == goal_loc:
            return [start_loc]
        if unit_kind == 'Army':
            adj_graph, dist_table = self.graph_army, self.dist_army
        else:
            adj_graph, dist_table = self.graph_fleet, self.dist_fleet
        if start_loc not in adj_graph or goal_loc not in dist_table.get(start_loc, {}):
            return None
        start_h = dist_table[start_loc].get(goal_loc, float('inf'))
        if start_h == float('inf'):
            return None

        open_heap = [(start_h, 0.0, start_loc)]
        g_scores = {start_loc: 0.0}
        parents = {}
        closed_set = set()
        expanded = 0
        while open_heap:
            if time.perf_counter() > deadline or expanded > self.MAX_NODES_PER_UNIT:
                return None
            f, g, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            closed_set.add(current)
            expanded += 1
            if current == goal_loc:
                path = [current]
                while current in parents:
                    current = parents[current]
                    path.append(current)
                return path[::-1]
            for nb in adj_graph.neighbors(current):
                if nb in closed_set:
                    continue
                if nb in friendly_occ and nb != goal_loc:
                    continue                       # cannot plan through own units
                step_cost = 1.0
                if nb in enemy_occ:
                    if nb == goal_loc:
                        step_cost += self.ENEMY_PENALTY_ATTACK
                    elif self.BLOCK_ENEMY_TRANSIT:
                        continue
                    else:
                        step_cost += self.ENEMY_PENALTY_PASS
                tentative_g = g + step_cost
                if tentative_g < g_scores.get(nb, float('inf')):
                    g_scores[nb] = tentative_g
                    parents[nb] = current
                    h = dist_table[nb].get(goal_loc, float('inf'))
                    if h == float('inf'):
                        continue
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, nb))
        return None

    # ------------------------------------------------ reactive supports
    def add_simple_supports(self, final_orders, unit_options):
        """After moves are chosen, convert another unit's order into a support
        for a move if a matching legal support order exists."""
        order_by_token = {}
        for order in final_orders:
            parts = order.split()
            if len(parts) >= 2:
                order_by_token[parts[0] + ' ' + parts[1]] = order
        move_tokens = [(t, o) for t, o in order_by_token.items()
                       if len(o.split()) >= 4 and o.split()[2] == '-']
        move_tokens.sort(key=lambda x: 0 if x[1].split()[3].upper() in self.supply_centers else 1)
        used = set()
        for target_token, target_order in move_tokens:
            for support_token, support_order in list(order_by_token.items()):
                if support_token == target_token or support_token in used:
                    continue
                if ' S ' in support_order:
                    continue
                for opt in unit_options.get(support_token, []):
                    if ' S ' in opt and opt.endswith(target_order):
                        order_by_token[support_token] = opt
                        used.add(support_token)
                        break
                if support_token in used:
                    break
        return list(order_by_token.values())

    # -------------------------------------------------- non-movement phases
    def handle_non_movement(self):
        phase_type = getattr(self.game, 'phase_type', 'M')
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = []
        if phase_type == 'R':
            friendly_occ, enemy_occ = self.get_dynamic_costs()
            my_centers = {c for c, p in self.get_center_owners().items()
                          if p == self.power_name}
        else:
            friendly_occ, enemy_occ, my_centers = set(), set(), set()

        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue
            strings = [o.as_string() if hasattr(o, 'as_string') else str(o) for o in possible]
            if phase_type == 'R':
                retreats = [s for s in strings if ' R ' in s]
                if retreats:
                    best_retreat, best_score = None, float('inf')
                    for s in retreats:
                        parts = s.split()
                        if len(parts) < 4:
                            continue
                        dest = parts[3].upper()
                        table = self.dist_fleet if parts[0] == 'F' else self.dist_army
                        score = 0.0
                        if dest in enemy_occ:
                            score += 1000.0
                        if dest in friendly_occ:
                            score += 500.0
                        if dest in table and my_centers:
                            best_dist = min(table[dest].get(c, float('inf')) for c in my_centers)
                            if best_dist != float('inf'):
                                score += best_dist
                        if score < best_score:
                            best_score, best_retreat = score, s
                    power_orders.append(best_retreat or random.choice(retreats))
                else:
                    disbands = [s for s in strings if s.split() and s.split()[-1] == 'D']
                    power_orders.append(disbands[0] if disbands else random.choice(strings))
            elif phase_type == 'A':
                builds = [s for s in strings if s.split() and s.split()[-1] == 'B']
                if builds:
                    armies = [s for s in builds if s.startswith('A ')]
                    fleets = [s for s in builds if s.startswith('F ')]
                    power_orders.append(random.choice(armies) if armies else random.choice(fleets))
                else:
                    disbands = [s for s in strings if s.split() and s.split()[-1] == 'D']
                    power_orders.append(disbands[0] if disbands else random.choice(strings))
            else:
                power_orders.append(random.choice(strings))
        return power_orders

    # ---------------------------------------------------- movement phases
    def _unit_options(self):
        all_possible_orders = self.game.get_all_possible_orders()
        unit_options = defaultdict(list)
        for loc in self.game.get_orderable_locations(self.power_name):
            for o in all_possible_orders.get(loc, []):
                s = o.as_string() if hasattr(o, 'as_string') else str(o)
                parts = s.split()
                if len(parts) >= 2:
                    unit_options[parts[0] + ' ' + parts[1]].append(s)
        return unit_options

    @staticmethod
    def _move_to(options, next_step):
        for opt in options:
            p = opt.split()
            if len(p) >= 4 and p[2] == '-' and p[3].upper() == next_step:
                return opt
        return None

    def _greedy_fallback(self, loc, options, dist_table, friendly_occ, enemy_occ,
                         reserved_dests, enemy_malus):
        best_order, best_score = None, -float('inf')
        for opt in options:
            parts = opt.split()
            if len(parts) < 4 or parts[2] != '-':
                continue
            dest = parts[3].upper()
            if dest in reserved_dests or dest in friendly_occ:
                continue
            curr_min = min(dist_table[loc].values()) if dist_table.get(loc) else float('inf')
            dest_min = (min(dist_table[dest].values())
                        if dest in dist_table and dist_table.get(dest) else float('inf'))
            score = curr_min - dest_min
            if dest in self.supply_centers:
                score += 50
            if dest in enemy_occ:
                score -= enemy_malus
            if score > best_score:
                best_score, best_order = score, opt
        return best_order

    def rank_targets(self, loc, kind, friendly_occ, enemy_occ, reserved_dests,
                     center_owners):
        return self.choose_targets(loc, kind, friendly_occ, enemy_occ,
                                   reserved_dests, center_owners)

    # Differences between the basic agent and Technique 1 in the move loop.
    RESERVE_CHOSEN_TARGET = True   # basic: reserve the SC a unit is heading to
    SKIP_RESERVED_IN_LOOP = False  # Technique 1 skips reserved targets instead
    FALLBACK_ENEMY_MALUS = 5       # basic fallback penalises enemy provinces

    def get_actions(self):
        deadline = time.perf_counter() + self.TIME_BUDGET
        if getattr(self.game, 'phase_type', 'M') != 'M':
            return self.handle_non_movement()
        try:
            unit_options = self._unit_options()
        except Exception:
            return []
        if not unit_options:
            return []

        friendly_occ, enemy_occ = self.get_dynamic_costs()
        center_owners = self.get_center_owners()
        final_orders, reserved_dests = [], set()

        # Units with the fewest legal options plan first.
        for token in sorted(unit_options, key=lambda t: len(unit_options[t])):
            options = unit_options[token]
            hold_opt = next((o for o in options if o.split()[-1].upper() == 'H'), None)
            default = hold_opt if hold_opt else options[0]
            if time.perf_counter() > deadline:
                final_orders.append(default)
                continue
            prefix, loc = token.split()[:2]
            loc = loc.upper()
            kind = 'Fleet' if prefix == 'F' else 'Army'
            dist_table = self.dist_army if kind == 'Army' else self.dist_fleet
            if loc not in dist_table:
                final_orders.append(default)
                continue

            targets = self.rank_targets(loc, kind, friendly_occ, enemy_occ,
                                        reserved_dests, center_owners)
            chosen, chosen_target = None, None
            for target in targets:
                if time.perf_counter() > deadline:
                    break
                if self.SKIP_RESERVED_IN_LOOP and target in reserved_dests:
                    continue
                path = self.astar_search(loc, target, kind, friendly_occ, enemy_occ, deadline)
                if path and len(path) > 1:
                    order = self._move_to(options, path[1].upper())
                    if order:
                        chosen, chosen_target = order, target
                        break

            if chosen is None:
                chosen = self._greedy_fallback(loc, options, dist_table, friendly_occ,
                                               enemy_occ, reserved_dests,
                                               self.FALLBACK_ENEMY_MALUS)
                chosen_target = None
            if chosen is None:
                final_orders.append(default)
                continue
            final_orders.append(chosen)
            dest = chosen.split()[3].upper()
            reserved_dests.add(dest)
            if self.RESERVE_CHOSEN_TARGET and chosen_target:
                reserved_dests.add(chosen_target)
            friendly_occ.discard(loc)
            friendly_occ.add(dest)

        return self.add_simple_supports(final_orders, unit_options)


###############################################################################
# Technique 1 (not called by the game): context-aware target selection.
# Same A* search and move loop; only the choice of targets changes.
###############################################################################
class Technique1Agent(BasicAStarAgent):
    MAX_TARGETS = 3
    BLOCK_ENEMY_TRANSIT = False   # enemy transit is penalised (+15), not banned
    RESERVE_CHOSEN_TARGET = False
    SKIP_RESERVED_IN_LOOP = True
    FALLBACK_ENEMY_MALUS = 0

    def __init__(self, agent_name="Technique1Agent"):
        super().__init__(agent_name)

    def _collect_supply_centers(self, game, locations_dict):
        sc_set = set()
        for sc in getattr(game.map, 'scs', []):
            name = str(getattr(sc, 'name', sc)).upper()
            if name in locations_dict:
                sc_set.add(name)
            base = name.split('/')[0]
            if base in locations_dict:
                sc_set.add(base)
        return sorted(sc_set)

    def rank_targets(self, loc, kind, friendly_occ, enemy_occ, reserved_dests,
                     center_owners):
        """Exclude centres we own, slightly prefer neutral centres, and penalise
        enemy-occupied centres. Falls back to plain distance if we own them all."""
        dist_table = self.dist_army if kind == 'Army' else self.dist_fleet
        scored = []
        for sc, d in dist_table[loc].items():
            if d == float('inf') or sc in friendly_occ:
                continue
            owner = center_owners.get(sc)
            if owner == self.power_name:
                continue
            score = float(d)
            if owner is None:
                score -= 1.0
            if sc in enemy_occ:
                score += 5.0
            scored.append((score, sc))
        if not scored:
            scored = [(float(d), sc) for sc, d in dist_table[loc].items()
                      if d != float('inf') and sc not in friendly_occ]
        scored.sort(key=lambda x: x[0])
        return [sc for _, sc in scored[:self.MAX_TARGETS_PER_UNIT]]
