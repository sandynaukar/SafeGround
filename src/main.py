import time
from src.airport_graph import AirportGraphManager
from src.symbolic_guardrail import SymbolicSafetyGuardrail
from src.llm_agent import LLMCognitiveAgent

def run_simulation_demo():
    print("=" * 80)
    print("          SAFEGROUND-LLM: NEURO-SYMBOLIC DECISION SUPPORT DEMO          ")
    print("=" * 80)
    
    # 1. Initialize System Components
    gm = AirportGraphManager()
    guardrail = SymbolicSafetyGuardrail(gm)
    agent = LLMCognitiveAgent(gm)
    
    print("\n[INIT] Airport topological map loaded.")
    print(f"       Nodes: {list(gm.graph.nodes)}")
    print(f"       Edges: {len(gm.graph.edges)} active taxiway segments.")
    
    # =========================================================================
    # SCENARIO 1: Heavy Aircraft Wingtip Clearance Check & Replanning
    # =========================================================================
    print("\n" + "-" * 50)
    print("SCENARIO 1: Heavy Aircraft (Boeing 777-300ER) Taxiing from Gate A1")
    print("           (Target: Runway 28L, Wingspan: 64.8m)")
    print("-" * 50)
    
    aircraft_callsign = "UAL901"
    aircraft_type = "B77W"
    wingspan = 64.8  # B777-300ER wingspan in meters
    start_gate = "Gate_A1"
    destination = "Hold_Short_28L"
    weather = "KSFO 081250Z 27015KT 10SM CLR 18/12 A2992"
    
    # First attempt: LLM Cognitive Agent proposes a routing
    proposal = agent.propose_clearance(
        aircraft_callsign=aircraft_callsign,
        aircraft_type=aircraft_type,
        wingspan=wingspan,
        start=start_gate,
        destination=destination,
        weather=weather,
        is_runway_cleared=False,
        attempt=1
    )
    
    print(f"\n[LLM PROPOSAL 1] Path: {proposal['proposed_path']}")
    print(f"                Phraseology: \"{proposal['icao_phraseology']}\"")
    print(f"                Rationale: {proposal['rationale']}")
    
    # Verify using Symbolic Guardrail
    is_safe, violations = guardrail.verify_taxi_path(
        aircraft_callsign=aircraft_callsign,
        wingspan=wingspan,
        proposed_path=proposal["proposed_path"],
        runway_cleared=False
    )
    
    if not is_safe:
        print("\n[GUARDRAIL STATUS] REJECTED - Safety Violations Found!")
        # Trigger LLM Reflection & Replanning Loop
        corrected_proposal = agent.reflect_and_replan(
            aircraft_callsign=aircraft_callsign,
            aircraft_type=aircraft_type,
            wingspan=wingspan,
            start=start_gate,
            destination=destination,
            weather=weather,
            is_runway_cleared=False,
            violation_messages=violations
        )
        
        # Verify the corrected path
        is_safe_2, violations_2 = guardrail.verify_taxi_path(
            aircraft_callsign=aircraft_callsign,
            wingspan=wingspan,
            proposed_path=corrected_proposal["proposed_path"],
            runway_cleared=False
        )
        
        if is_safe_2:
            print("\n[GUARDRAIL STATUS] APPROVED - Corrected path passes all safety constraints.")
            print(f"[FINAL CLEARANCE] Path: {corrected_proposal['proposed_path']}")
            print(f"                  Phraseology: \"{corrected_proposal['icao_phraseology']}\"")
            print(f"                  Explanation: {corrected_proposal['rationale']}")
        else:
            print(f"\n[GUARDRAIL STATUS] REJECTED AGAIN: {violations_2}")
            
    # =========================================================================
    # SCENARIO 2: Runway Incursion Prevention (No Active Runway Crossing Cleared)
    # =========================================================================
    print("\n" + "-" * 50)
    print("SCENARIO 2: Runway Incursion Prevention (A320, calls Rwy_28L_Entry without crossing clearance)")
    print("-" * 50)
    
    aircraft_callsign = "AAL424"
    aircraft_type = "A320"
    wingspan = 35.8
    start_gate = "Gate_A1"
    destination = "Rwy_28L_Entry" # Destination is runway entry, but crossing is NOT cleared
    
    # First attempt: LLM proposes direct taxi onto the runway
    proposal = agent.propose_clearance(
        aircraft_callsign=aircraft_callsign,
        aircraft_type=aircraft_type,
        wingspan=wingspan,
        start=start_gate,
        destination=destination,
        weather=weather,
        is_runway_cleared=False, # RUNWAY CROSSING NOT CLEARED
        attempt=1
    )
    
    print(f"\n[LLM PROPOSAL 1] Path: {proposal['proposed_path']}")
    print(f"                Phraseology: \"{proposal['icao_phraseology']}\"")
    print(f"                Rationale: {proposal['rationale']}")
    
    # Verify using Symbolic Guardrail
    is_safe, violations = guardrail.verify_taxi_path(
        aircraft_callsign=aircraft_callsign,
        wingspan=wingspan,
        proposed_path=proposal["proposed_path"],
        runway_cleared=False
    )
    
    if not is_safe:
        print("\n[GUARDRAIL STATUS] REJECTED - Safety Violations Found!")
        # Trigger LLM Reflection & Replanning Loop (shortening path to Hold Short)
        corrected_proposal = agent.reflect_and_replan(
            aircraft_callsign=aircraft_callsign,
            aircraft_type=aircraft_type,
            wingspan=wingspan,
            start=start_gate,
            destination="Hold_Short_28L", # Replanning target
            weather=weather,
            is_runway_cleared=False,
            violation_messages=violations
        )
        
        # Verify the corrected path
        is_safe_2, violations_2 = guardrail.verify_taxi_path(
            aircraft_callsign=aircraft_callsign,
            wingspan=wingspan,
            proposed_path=corrected_proposal["proposed_path"],
            runway_cleared=False
        )
        
        if is_safe_2:
            print("\n[GUARDRAIL STATUS] APPROVED - Corrected path holds short of active runways.")
            print(f"[FINAL CLEARANCE] Path: {corrected_proposal['proposed_path']}")
            print(f"                  Phraseology: \"{corrected_proposal['icao_phraseology']}\"")
            print(f"                  Explanation: {corrected_proposal['rationale']}")
        else:
            print(f"\n[GUARDRAIL STATUS] REJECTED AGAIN: {violations_2}")

    print("\n" + "=" * 80)
    print("                  DEMO COMPLETED SUCCESSFULLY                  ")
    print("=" * 80)

if __name__ == "__main__":
    run_simulation_demo()
