import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from src.airport_graph import AirportGraphManager

# Load environment variables from .env if present
load_dotenv()

class LLMCognitiveAgent:
    """
    Cognitive Reasoning Agent for SafeGround-LLM.
    
    Supports:
    1. Real LLM integration using Gemini 1.5 Flash/Pro with automatic tool calling.
    2. Fallback simulation mode if GEMINI_API_KEY is not set.
    3. Self-reflection loops to replan when safety violations are flagged.
    """
    def __init__(self, graph_manager: AirportGraphManager, model_name: str = "gemini-1.5-flash"):
        self.gm = graph_manager
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.use_real_llm = bool(self.api_key)
        self.model_name = model_name
        
        if self.use_real_llm:
            genai.configure(api_key=self.api_key)
            self._setup_real_model()
        else:
            print("[LLM AGENT] GEMINI_API_KEY not found. Running agent in Simulation Fallback mode.")

    def _setup_real_model(self):
        """
        Registers local tools with the Gemini API and prepares the model.
        """
        # Expose graph querying methods as tools for the model
        def get_shortest_path(start: str, end: str) -> str:
            """Gets the basic shortest topological route between two nodes on the airport map."""
            path = self.gm.get_shortest_path(start, end)
            return json.dumps(path)

        def check_segment_status(u: str, v: str) -> str:
            """Checks safety attributes of a specific taxiway segment u -> v (like max wingspan support)."""
            attrs = self.gm.check_edge_safety_attributes(u, v)
            return json.dumps(attrs)

        self.tools = [get_shortest_path, check_segment_status]
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=self.tools,
            generation_config={"response_mime_type": "application/json"}
        )

    def propose_clearance(self, 
                          aircraft_callsign: str, 
                          aircraft_type: str, 
                          wingspan: float,
                          start: str, 
                          destination: str, 
                          weather: str,
                          is_runway_cleared: bool,
                          attempt: int = 1,
                          reflection_feedback: str = "") -> Dict:
        """
        Proposes a taxi clearance. Automatically uses Gemini API if key is available,
        otherwise uses mock logic.
        """
        if self.use_real_llm:
            return self._propose_clearance_real_llm(
                aircraft_callsign, aircraft_type, wingspan, start, destination, weather, is_runway_cleared, attempt, reflection_feedback
            )
        else:
            return self._propose_clearance_mock(
                aircraft_callsign, aircraft_type, wingspan, start, destination, weather, is_runway_cleared, attempt
            )

    def _propose_clearance_real_llm(self,
                                    aircraft_callsign: str,
                                    aircraft_type: str,
                                    wingspan: float,
                                    start: str,
                                    destination: str,
                                    weather: str,
                                    is_runway_cleared: bool,
                                    attempt: int,
                                    reflection_feedback: str) -> Dict:
        """
        Queries Gemini with automatic function calling enabled to build a safe path.
        Includes safety try-except blocks to catch key restrictions or API issues.
        """
        prompt = f"""
        You are SafeGround-LLM, an expert airport ground control decision-support agent.
        Your goal is to suggest a safe, efficient taxi path and write standard ATC clearance phraseology.

        [Flight Details]
        - Callsign: {aircraft_callsign}
        - Type: {aircraft_type}
        - Wingspan: {wingspan} meters
        - Start Node: {start}
        - Destination Node: {destination}
        - Weather: {weather}
        - Active Runway Crossing Approved: {is_runway_cleared}

        [Routing Rules]
        1. Query the shortest path using `get_shortest_path`.
        2. Query individual segments using `check_segment_status` to ensure the segment's `max_wingspan` is greater than or equal to {wingspan} meters.
        3. If a segment is too narrow, find an alternative route.
        4. If `Active Runway Crossing Approved` is False, you MUST NOT include any node type 'runway_entry' in your proposed path. You must stop and Hold Short at the corresponding 'hold_short' node.

        [JSON Schema Output]
        Provide your final recommendation in this JSON format:
        {{
          "proposed_path": ["node_1", "node_2", ...],
          "icao_phraseology": "Standard ICAO command string",
          "rationale": "Detailed logical explanation of why this path was chosen, listing safety checks."
        }}
        """

        if attempt > 1 and reflection_feedback:
            prompt += f"\n\n[ATTEMPT {attempt} - SELF CORRECTION REQUIRED]\n"
            prompt += f"Your previous proposal was REJECTED by the Symbolic Safety Guardrail for the following reasons:\n"
            prompt += f"{reflection_feedback}\n"
            prompt += "Please call your tools again to find a route that resolves these violations."

        try:
            # Start a chat with automatic function calling enabled
            chat = self.model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(prompt)
            result = json.loads(response.text)
            # Add metadata
            result["metadata"] = {
                "aircraft": aircraft_callsign,
                "type": aircraft_type,
                "wingspan_m": wingspan,
                "weather": weather,
                "attempt": attempt,
                "engine": f"Gemini ({self.model_name})"
            }
            return result
        except Exception as e:
            # Catch API errors (e.g. key invalid, project not configured, rate limits)
            print(f"[LLM AGENT ERROR] Gemini API Call failed: {str(e)}")
            return {
                "proposed_path": [start],
                "icao_phraseology": f"{aircraft_callsign}, hold position.",
                "rationale": f"API ERROR: Failed to run Gemini model query.",
                "metadata": {
                    "aircraft": aircraft_callsign,
                    "type": aircraft_type,
                    "wingspan_m": wingspan,
                    "weather": weather,
                    "attempt": attempt,
                    "engine": "Simulation Fallback (Error Mode)",
                    "error": str(e)
                }
            }

    def _propose_clearance_mock(self, 
                                aircraft_callsign: str, 
                                aircraft_type: str, 
                                wingspan: float,
                                start: str, 
                                destination: str, 
                                weather: str,
                                is_runway_cleared: bool,
                                attempt: int = 1) -> Dict:
        """
        Dynamic mock generator that works with both mock and real OpenStreetMap graphs.
        """
        # Find the actual shortest path in the graph
        path = self.gm.get_shortest_path(start, destination)
        
        # Fallback if no path is found
        if not path:
            path = [start, destination] if start in self.gm.graph and destination in self.gm.graph else [start]
            
        proposed_path = list(path)
        
        # Simulate LLM path mistakes for Attempt 1
        if attempt == 1:
            # 1. Wingspan violation simulation:
            # If wingspan > 50.0 (heavy), and we are at a mock airport,
            # force path to start at Gate_A1 (which has a 36m limit edge) even if Gate_A2 was selected.
            if wingspan > 50.0 and start == "Gate_A2" and "Gate_A1" in self.gm.graph:
                alt_path = self.gm.get_shortest_path("Gate_A1", destination)
                if alt_path:
                    proposed_path = alt_path
            
            # 2. Runway crossing simulation:
            # Under Attempt 1, the mock clearance goes all the way onto the runway (runway_entry)
            # even if runway crossing is not cleared, to trigger the guardrail.
            pass
            
        else: # Attempt 2 (self-correction reflection)
            # 1. Resolve wingspan violation:
            if wingspan > 50.0 and start == "Gate_A1" and "Gate_A2" in self.gm.graph:
                alt_path = self.gm.get_shortest_path("Gate_A2", destination)
                if alt_path:
                    proposed_path = alt_path
                    
            # 2. Resolve runway crossing:
            if not is_runway_cleared:
                new_path = []
                for node in proposed_path:
                    n_type = self.gm.graph.nodes[node].get("type") if node in self.gm.graph else None
                    if n_type == "runway_entry":
                        break
                    new_path.append(node)
                
                if new_path:
                    proposed_path = new_path
                else:
                    proposed_path = [start]
                    
        # Generate clean ATC phraseology
        phraseology = self._generate_icao_phraseology(aircraft_callsign, proposed_path, is_runway_cleared)
        
        # Build dynamic explanation
        if attempt == 1:
            if wingspan > 50.0 and proposed_path[0] == "Gate_A1":
                rationale = f"Planned shortest route from Gate A1. Note: Gate A1 has taxiway width restrictions but offers 10% faster taxi time."
            elif not is_runway_cleared and self.gm.graph.has_node(proposed_path[-1]) and self.gm.graph.nodes[proposed_path[-1]].get("type") == "runway_entry":
                rationale = f"Direct route to runway entry point {proposed_path[-1]} to minimize takeoff queue delays, crossing hold-short line."
            else:
                rationale = f"Shortest direct route from {start} to {destination} using active taxiway segments."
        else:
            if wingspan > 50.0 and proposed_path[0] == "Gate_A2":
                rationale = f"Corrected route: Re-routed Heavy aircraft via stand Gate A2 and wider taxiway segments (65m limit) to ensure safe wingtip separation."
            elif not is_runway_cleared:
                hold_node = proposed_path[-1]
                rationale = f"Corrected route: Shortened route to hold short at {hold_node} because active runway crossing clearance is pending."
            else:
                rationale = f"Corrected alternative route bypassing closed/narrow segments to destination {destination}."

        return {
            "proposed_path": proposed_path,
            "icao_phraseology": phraseology,
            "rationale": rationale,
            "metadata": {
                "aircraft": aircraft_callsign,
                "type": aircraft_type,
                "wingspan_m": wingspan,
                "attempt": attempt,
                "engine": "Simulation Fallback (Dynamic)"
            }
        }

    def _generate_icao_phraseology(self, callsign: str, path: List[str], runway_cleared: bool) -> str:
        route_instructions = []
        for i in range(len(path) - 1):
            edge_name = self.gm.check_edge_safety_attributes(path[i], path[i+1]).get("name", "Unknown")
            if edge_name not in route_instructions and edge_name != "Unknown":
                route_instructions.append(edge_name)
        
        route_str = ", ".join(route_instructions)
        destination_node = path[-1]
        
        if "Rwy" in destination_node or "Entry" in destination_node:
            if runway_cleared:
                return f"{callsign}, taxi to {destination_node.replace('_Entry', '')} via {route_str}."
            else:
                return f"{callsign}, taxi to Hold Short {destination_node.replace('_Entry', '').replace('Rwy', 'Runway')} via {route_str}."
        else:
            return f"{callsign}, taxi to Hold Short {destination_node.replace('_Entry', '').replace('Rwy', 'Runway')} via {route_str}."

    def reflect_and_replan(self, 
                           aircraft_callsign: str, 
                           aircraft_type: str, 
                           wingspan: float,
                           start: str, 
                           destination: str, 
                           weather: str,
                           is_runway_cleared: bool,
                           violation_messages: List[str]) -> Dict:
        """
        Feeds the safety violations back into the LLM context to generate a safe alternative path.
        """
        print(f"\n[LLM REFLECTION] Re-routing {aircraft_callsign} due to safety violations:")
        for v in violation_messages:
            print(f"  -> VIOLATION: {v}")
            
        violation_feedback = "\n".join([f"- {v}" for v in violation_messages])
        
        if self.use_real_llm:
            return self.propose_clearance(
                aircraft_callsign=aircraft_callsign,
                aircraft_type=aircraft_type,
                wingspan=wingspan,
                start=start,
                destination=destination,
                weather=weather,
                is_runway_cleared=is_runway_cleared,
                attempt=2,
                reflection_feedback=violation_feedback
            )
        else:
            return self._propose_clearance_mock(
                aircraft_callsign=aircraft_callsign,
                aircraft_type=aircraft_type,
                wingspan=wingspan,
                start=start,
                destination=destination,
                weather=weather,
                is_runway_cleared=is_runway_cleared,
                attempt=2
            )
