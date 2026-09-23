import time
from diplomacy import Game
from agent_2 import StudentAgent
from agent_baselines import StaticAgent
from game import run_one_game

def test_single_phase():
    """Test 1: Does the agent initialize and produce legal moves in S1901M?"""
    print("=" * 50)
    print("TEST 1: Single Phase Legality Check")
    print("=" * 50)
    
    game = Game()
    agent = StudentAgent()
    power = "ENGLAND"
    
    try:
        # Initialize Agent
        start_init = time.perf_counter()
        agent.new_game(game, power)
        init_time = time.perf_counter() - start_init
        print(f"[OK] Agent initialized in {init_time:.4f}s")
        
        # Generate Actions
        start_act = time.perf_counter()
        orders = agent.get_actions()
        act_time = time.perf_counter() - start_act
        print(f"[OK] Generated {len(orders)} orders in {act_time:.4f}s")
        
        # Display Orders
        print("\nGenerated Orders:")
        for o in sorted(orders):
            print(f"  {o}")
            
        # Verify Legality
        all_possible = game.get_all_possible_orders()
        valid_set = set()
        
        # Build reference set of legal strings for this power
        my_units_str = game.powers[power].units
        for unit_str in my_units_str:
            loc = unit_str.split()[1]
            if loc in all_possible:
                for o_obj in all_possible[loc]:
                    s = o_obj.as_string() if hasattr(o_obj, 'as_string') else str(o_obj)
                    valid_set.add(s)
        
        illegal_found = False
        for o in orders:
            if o not in valid_set:
                print(f"\n[FAIL] ILLEGAL ORDER DETECTED: {o}")
                illegal_found = True
        
        if not illegal_found:
            print("\n[PASS] All generated orders are LEGAL.")
        else:
            print("\n[FAIL] Some orders were illegal.")
            return False
            
    except Exception as e:
        print(f"\n[CRASH] Error during single phase test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def test_quick_game():
    """Test 2: Does the agent survive a short game (up to 1902)?"""
    print("\n" + "=" * 50)
    print("TEST 2: Quick Game Stability Check (End Year 1902)")
    print("=" * 50)
    
    ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
    
    def make_agents(player_power="ENGLAND"):
        agents = {}
        for p in ALL_POWERS:
            if p == player_power:
                agents[p] = StudentAgent()
            else:
                agents[p] = StaticAgent()
        return agents

    try:
        agents = make_agents("ENGLAND")
        results, year = run_one_game(agents, end_year=1902)
        
        print(f"\n[PASS] Game completed successfully up to year {year}.")
        print("Final Supply Centers:")
        for p, sc in sorted(results.items()):
            marker = " <-- PLAYER" if p == "ENGLAND" else ""
            print(f"  {p}: {sc}{marker}")
            
        if results["ENGLAND"] >= 3:
            print("\n[INFO] England maintained or gained territory.")
        else:
            print("\n[WARN] England lost territory (expected against Static if bugs exist, or just bad luck).")
            
        return True
        
    except Exception as e:
        print(f"\n[CRASH] Error during quick game: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success_1 = test_single_phase()
    
    if success_1:
        success_2 = test_quick_game()
    else:
        success_2 = False
        print("\nSkipping Quick Game Test because Single Phase Test Failed.")
        
    print("\n" + "=" * 50)
    if success_1 and success_2:
        print("ALL TESTS PASSED. Your A* Agent is ready for benchmarking.")
    else:
        print("SOME TESTS FAILED. Please check the output above.")
    print("=" * 50)