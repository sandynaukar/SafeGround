import networkx as nx
from typing import Dict, List, Tuple
import math
import requests

class AirportGraphManager:
    """
    Manages the airport surface layout represented as a topological directed graph.
    Vertices (Nodes) are physical points (Gates, Intersections, Hold-short bars, Runways).
    Edges are segments (Taxiways, Runway segments).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_mock_airport()

    def _build_mock_airport(self):
        """
        Builds a mock airport representation with physical 2D layout coordinates (pos).
        """
        # Add Nodes: (id, attributes)
        # Gates
        self.graph.add_node("Gate_A1", type="gate", pos=(2.0, 10.0), description="Terminal 1 Gate A1 (Light/Medium)", elevation=2.0)
        self.graph.add_node("Gate_A2", type="gate", pos=(4.0, 10.0), description="Terminal 1 Gate A2 (Heavy Capable)", elevation=2.0)
        
        # Taxiway Intersections
        self.graph.add_node("Int_A_B", type="intersection", pos=(3.0, 8.0), description="Intersection of TWY Alpha and TWY Bravo", elevation=2.0)
        self.graph.add_node("Int_B_C", type="intersection", pos=(3.0, 5.0), description="Intersection of TWY Bravo and TWY Charlie", elevation=2.0)
        self.graph.add_node("Int_B_D", type="intersection", pos=(5.0, 5.0), description="Intersection on parallel TWY Delta", elevation=2.0)
        
        # Hold Short Points
        self.graph.add_node("Hold_Short_28L", type="hold_short", pos=(3.0, 3.0), target_runway="Runway_28L", elevation=2.0)
        self.graph.add_node("Hold_Short_28R", type="hold_short", pos=(6.0, 3.0), target_runway="Runway_28R", elevation=2.0)
        
        # Runway Entry Points
        self.graph.add_node("Rwy_28L_Entry", type="runway_entry", pos=(3.0, 1.0), elevation=2.0)
        self.graph.add_node("Rwy_28R_Entry", type="runway_entry", pos=(6.0, 1.0), elevation=2.0)

        # Add Edges (Taxiways / Segments)
        # Gate_A1 (Light) to Intersections
        self.graph.add_edge("Gate_A1", "Int_A_B", name="Alpha", weight=100, active=True, max_wingspan=36.0)
        self.graph.add_edge("Int_A_B", "Gate_A1", name="Alpha", weight=100, active=True, max_wingspan=36.0)
        
        # Gate_A2 (Heavy Capable) to Intersections
        self.graph.add_edge("Gate_A2", "Int_A_B", name="Bravo_West", weight=120, active=True, max_wingspan=65.0)
        self.graph.add_edge("Int_A_B", "Gate_A2", name="Bravo_West", weight=120, active=True, max_wingspan=65.0)
        
        # Main Taxiway Bravo (Wide)
        self.graph.add_edge("Int_A_B", "Int_B_C", name="Bravo", weight=300, active=True, max_wingspan=65.0)
        self.graph.add_edge("Int_B_C", "Int_A_B", name="Bravo", weight=300, active=True, max_wingspan=65.0)
        
        # Narrow Taxiway Charlie (Max 52m - e.g. B767/A300, but not B777/A380)
        self.graph.add_edge("Int_B_C", "Hold_Short_28L", name="Charlie", weight=150, active=True, max_wingspan=52.0)
        
        # Parallel Wide Taxiway Delta (Max 65m - Heavy Capable)
        self.graph.add_edge("Int_B_C", "Int_B_D", name="Delta", weight=180, active=True, max_wingspan=65.0)
        self.graph.add_edge("Int_B_D", "Hold_Short_28L", name="Delta", weight=100, active=True, max_wingspan=65.0)
        
        # Crossing Runway / Linking to Hold Short 28R
        self.graph.add_edge("Rwy_28L_Entry", "Hold_Short_28R", name="Charlie_Cross", weight=200, active=True, max_wingspan=65.0)
        
        # Connect hold short bars to runway entries
        self.graph.add_edge("Hold_Short_28L", "Rwy_28L_Entry", name="Entry_28L", weight=10, active=True, max_wingspan=80.0)
        self.graph.add_edge("Hold_Short_28R", "Rwy_28R_Entry", name="Entry_28R", weight=10, active=True, max_wingspan=80.0)

    def get_shortest_path(self, start: str, end: str) -> List[str]:
        """
        Helper tool for the LLM to get the basic shortest topological path.
        """
        try:
            return nx.shortest_path(self.graph, source=start, target=end, weight="weight")
        except nx.NetworkXNoPath:
            return []

    def check_edge_safety_attributes(self, u: str, v: str) -> Dict:
        """
        Returns physical properties of a taxi segment.
        """
        if self.graph.has_edge(u, v):
            return self.graph[u][v]
        return {}

    def load_airport(self, airport_code: str):
        """
        Dynamically loads the airport layout graph from OpenStreetMap.
        Falls back to the mock graph if connection fails.
        """
        from src.live_traffic import OpenSkyTrafficFeed
        bounds = OpenSkyTrafficFeed.AIRPORT_BOUNDS.get(airport_code)
        if not bounds:
            self._build_mock_airport()
            return
            
        print(f"[GRAPH] Fetching geometry for {bounds['name']} ({airport_code}) from OpenStreetMap...")
        
        # Bounding box query is faster and more reliable
        query = f"""[out:json][timeout:15];
        (
          way["aeroway"~"runway|taxiway"]({bounds['lamin']},{bounds['lomin']},{bounds['lamax']},{bounds['lomax']});
          node(w);
        );
        out body;"""
        
        mirrors = [
            "https://lz4.overpass-api.de/api/interpreter",
            "https://z.overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass-api.de/api/interpreter"
        ]
        
        headers = {
            "User-Agent": "SafeGroundAirportGraphBuilder/1.0 (laxmi@gemini.api)"
        }
        
        data = None
        for url in mirrors:
            try:
                response = requests.get(url, params={"data": query}, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print(f"[GRAPH] Loaded OSM geometry from mirror: {url}")
                    break
            except Exception as e:
                print(f"[GRAPH WARNING] Mirror {url} failed: {e}")
                
        if not data:
            print("[GRAPH WARNING] Failed to download layout from OpenStreetMap. Falling back to mock layout.")
            self._build_mock_airport()
            return
            
        elements = data.get("elements", [])
        nodes = [e for e in elements if e["type"] == "node"]
        ways = [e for e in elements if e["type"] == "way"]
        
        if not nodes or not ways:
            print("[GRAPH WARNING] No aeroways found in OSM data. Falling back to mock layout.")
            self._build_mock_airport()
            return
            
        # Build raw undirected graph
        raw_G = nx.Graph()
        for node in nodes:
            raw_G.add_node(node["id"], 
                           pos=(node["lon"], node["lat"]), 
                           tags=node.get("tags", {}))
            
        for way in ways:
            tags = way.get("tags", {})
            name = tags.get("name", tags.get("ref", tags.get("aeroway", "unnamed")))
            aeroway_type = tags.get("aeroway", "taxiway")
            nodes_list = way["nodes"]
            for i in range(len(nodes_list) - 1):
                raw_G.add_edge(nodes_list[i], nodes_list[i+1], name=name, aeroway=aeroway_type)

        # Haversine distance calculator
        def haversine_distance(coord1, coord2):
            lon1, lat1 = coord1
            lon2, lat2 = coord2
            R = 6371000.0  # earth radius in meters
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            d_phi = math.radians(lat2 - lat1)
            d_lon = math.radians(lon2 - lon1)
            
            a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c

        # Identify nodes to keep
        nodes_to_keep = set()
        for node in raw_G.nodes:
            deg = raw_G.degree(node)
            tags = raw_G.nodes[node].get("tags", {})
            
            is_special = (
                deg != 2 or
                "gate" in tags.get("aeroway", "") or
                "holding_position" in tags.get("aeroway", "") or
                tags.get("aeroway") in ["gate", "holding_position"]
            )
            
            # Check if it connects runway and taxiway
            if not is_special and deg == 2:
                neighbors = list(raw_G.neighbors(node))
                edges_aeroway = [raw_G[node][nbr].get("aeroway") for nbr in neighbors]
                if "runway" in edges_aeroway and "taxiway" in edges_aeroway:
                    is_special = True
                    
            if is_special:
                nodes_to_keep.add(node)
                
        # Topological simplification
        simplified_G = raw_G.copy()
        for node in list(simplified_G.nodes):
            if node in nodes_to_keep:
                continue
                
            neighbors = list(simplified_G.neighbors(node))
            if len(neighbors) == 2:
                u, v = neighbors
                w1 = haversine_distance(simplified_G.nodes[u]["pos"], simplified_G.nodes[node]["pos"])
                w2 = haversine_distance(simplified_G.nodes[node]["pos"], simplified_G.nodes[v]["pos"])
                weight = w1 + w2
                
                name1 = simplified_G[u][node].get("name", "unnamed")
                type1 = simplified_G[u][node].get("aeroway", "taxiway")
                
                if type1 == "runway":
                    max_wingspan = 80.0
                else:
                    name_upper = name1.upper()
                    if "A" in name_upper or "ALPHA" in name_upper:
                        max_wingspan = 36.0
                    elif "C" in name_upper or "CHARLIE" in name_upper:
                        max_wingspan = 52.0
                    else:
                        max_wingspan = 65.0
                
                simplified_G.remove_node(node)
                simplified_G.add_edge(u, v, name=name1, weight=weight, active=True, max_wingspan=max_wingspan, aeroway=type1)

        # Filter largest connected component
        if simplified_G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(simplified_G), key=len)
            simplified_G = simplified_G.subgraph(largest_cc).copy()
            
        # Re-initialize clean directed graph
        self.graph = nx.DiGraph()
        rename_map = {}
        gate_cnt = 1
        hs_cnt = 1
        rwy_cnt = 1
        int_cnt = 1
        
        # Rename nodes to nice clean identifiers
        for node in simplified_G.nodes:
            tags = simplified_G.nodes[node].get("tags", {})
            pos = simplified_G.nodes[node]["pos"]
            
            is_gate = "gate" in tags.get("aeroway", "") or tags.get("aeroway") == "gate"
            is_hs = "holding_position" in tags.get("aeroway", "") or tags.get("aeroway") == "holding_position"
            
            # Check if node touches a runway edge
            neighbors = list(simplified_G.neighbors(node))
            edge_types = [simplified_G[node][nbr].get("aeroway") for nbr in neighbors]
            is_rwy = "runway" in edge_types
            
            if is_gate:
                new_name = f"Gate_G{gate_cnt}"
                gate_cnt += 1
                n_type = "gate"
                desc = f"Terminal Gate Stand G{gate_cnt-1}"
            elif is_hs:
                new_name = f"Hold_Short_{hs_cnt}"
                hs_cnt += 1
                n_type = "hold_short"
                desc = f"Runway Hold Short Position {hs_cnt-1}"
            elif is_rwy:
                new_name = f"Rwy_Entry_{rwy_cnt}"
                rwy_cnt += 1
                n_type = "runway_entry"
                desc = f"Runway Entry/Exit Point {rwy_cnt-1}"
            else:
                new_name = f"Int_{int_cnt}"
                int_cnt += 1
                n_type = "intersection"
                desc = f"Taxiway Junction {int_cnt-1}"
                
            rename_map[node] = new_name
            self.graph.add_node(new_name, type=n_type, pos=pos, description=desc, elevation=None)
            
        # Add bidirected edges
        for u, v, attrs in simplified_G.edges(data=True):
            new_u = rename_map[u]
            new_v = rename_map[v]
            self.graph.add_edge(new_u, new_v, **attrs)
            self.graph.add_edge(new_v, new_u, **attrs)
            
        print(f"[GRAPH] Loaded real graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        
        # Load elevations
        self._fetch_elevations_for_graph()

    def _fetch_elevations_for_graph(self):
        """
        Fetches elevations for all nodes in the graph in a single batch query.
        """
        nodes = list(self.graph.nodes)
        if not nodes:
            return
            
        lats = [str(self.graph.nodes[n]["pos"][1]) for n in nodes]
        lons = [str(self.graph.nodes[n]["pos"][0]) for n in nodes]
        
        # Batch coordinates using commas
        lat_str = ",".join(lats)
        lon_str = ",".join(lons)
        
        url = "https://api.open-meteo.com/v1/elevation"
        try:
            response = requests.get(url, params={"latitude": lat_str, "longitude": lon_str}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                elevations = data.get("elevation", [])
                for i, n in enumerate(nodes):
                    if i < len(elevations):
                        self.graph.nodes[n]["elevation"] = elevations[i]
                print(f"[GRAPH] Successfully loaded elevations (terrain data) for {len(elevations)} nodes.")
        except Exception as e:
            print(f"[GRAPH WARNING] Failed to fetch node elevations: {e}")
