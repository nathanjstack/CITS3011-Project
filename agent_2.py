import time
import random
import networkx as nx
from collections import defaultdict
from agent_baselines import Agent

INF = float("inf")


class StudentAgent(Agent):
    """
    Technique 2 (joint-order planning)

    Pipeline per movement phase:
      goal assignment -> internal adjudicator -> local search over joint orders
      (attack+support and convoy macros), evaluated against a single enemy
      model in which every enemy unit holds.
    """

    def __init__(self, agent_name="SearchAgent"):
        super().__init__(agent_name)
        self.game = None
        self.power_name = None
        self.graph_army = None
        self.graph_fleet = None
        self.dist_army = {}
        self.dist_fleet = {}
        self.all_dist_army = {}
        self.all_dist_fleet = {}
        self.supply_centers = []
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
        self.LEADER_SC = 12             # owners with >= this many SCs get extra attention

    # ------------------------------------------------------------------ setup
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self._dcache = {}
        self.home_centers = {str(c).upper() for c in game.get_centers(power_name)}
        self.build_static_map(game)

    def update_game(self, all_power_orders):
        # Do not change this code.
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
        # Single enemy model: every relevant enemy unit holds.
        e_hold = [('H',)] * len(rel)
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

        def raw_score(my_orders):
            total = 0.0
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
            return total

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

        start_plan = [b[0] for b in base]
        start_str = [b[1] for b in base]
        best, best_plan, best_str = search(start_plan, start_str, deadline, self.MAX_RESTARTS)
        return best_str

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
                return orders
        try:
            deadline = t0 + self.SEARCH_BUDGET
            return list(self.plan_movement(unit_options, deadline))
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