from typing import List, Dict, Tuple
from src.airport_graph import AirportGraphManager

class SymbolicSafetyGuardrail:
    """
    A deterministic rule-based verification engine that checks proposed LLM clearances 
    against physical and operational safety constraints.
    """
    def __init__(self, graph_manager: AirportGraphManager):
        self.gm = graph_manager
        # Track simulated closed segments (e.g., due to construction or FOD)
        self.closed_segments = set()
        # Track simulated occupied segments (e.g., another aircraft taxiing)
        self.occupied_segments = set() # Set of Tuple[str, str]

    def close_segment(self, u: str, v: str):
        self.closed_segments.add((u, v))

    def occupy_segment(self, u: str, v: str):
        self.occupied_segments.add((u, v))

    def clear_segment(self, u: str, v: str):
        self.closed_segments.discard((u, v))
        self.occupied_segments.discard((u, v))

    def verify_taxi_path(self, 
                         aircraft_callsign: str, 
                         wingspan: float, 
                         proposed_path: List[str], 
                         runway_cleared: bool = False) -> Tuple[bool, List[str]]:
        """
        Verifies a proposed taxi path.
        Returns:
            - Tuple[bool, List[str]]: (Is_Safe, List of Rule Violation Messages)
        """
        violations = []

        # 1. Path Continuity Check
        for i in range(len(proposed_path) - 1):
            u, v = proposed_path[i], proposed_path[i+1]
            if not self.gm.graph.has_edge(u, v):
                violations.append(f"Invalid edge transition: No physical taxi link exists between {u} and {v}.")
                continue

            # 2. Wingspan Limit Check
            edge_data = self.gm.check_edge_safety_attributes(u, v)
            max_wingspan = edge_data.get("max_wingspan", 0.0)
            if wingspan > max_wingspan:
                violations.append(
                    f"Wingspan violation on segment {u}->{v} ({edge_data.get('name')}). "
                    f"Aircraft wingspan is {wingspan}m, but segment limit is {max_wingspan}m."
                )

            # 3. Dynamic Hazard Check (Closed or Construction Segments)
            if (u, v) in self.closed_segments or (v, u) in self.closed_segments:
                violations.append(f"Segment {u}->{v} ({edge_data.get('name')}) is currently CLOSED due to operations.")

            # 4. Separation / Separation Violation Check (Friction / Collisions)
            if (u, v) in self.occupied_segments or (v, u) in self.occupied_segments:
                violations.append(f"Separation conflict on segment {u}->{v}: Blocked by traffic.")

            # 5. Runway Incursion / Clearance Check
            v_type = self.gm.graph.nodes[v].get("type")
            if v_type == "runway_entry" and not runway_cleared:
                violations.append(
                    f"Runway Incursion Risk at {v}. "
                    f"Proposed path crosses hold-short point without active Runway Clearance."
                )

        is_safe = len(violations) == 0
        return is_safe, violations
