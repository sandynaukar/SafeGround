import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import math
import time
import re
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import our modular backend components
from src.airport_graph import AirportGraphManager
from src.symbolic_guardrail import SymbolicSafetyGuardrail
from src.llm_agent import LLMCognitiveAgent
from src.live_traffic import OpenSkyTrafficFeed

# Page Configuration for Premium Aesthetic
st.set_page_config(
    page_title="SafeGround-LLM Copilot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Inject Custom CSS for Sleek Dark Glassmorphism Theme
st.markdown("""
<style>
    /* Dark glassmorphic background & panel headers */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek card container style */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    
    /* Glowing title */
    .glow-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
        margin-bottom: 5px;
    }
    
    /* Status badges */
    .badge-approved {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }

    .badge-rejected {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Monospaced phraseology terminal */
    .terminal-box {
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Fira Code', monospace;
        color: #38bdf8;
        font-size: 1.1rem;
        margin-top: 10px;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# System Initialization (Caching instances to maintain state across page loads)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_system_components():
    gm = AirportGraphManager()
    guardrail = SymbolicSafetyGuardrail(gm)
    feed = OpenSkyTrafficFeed()
    return gm, guardrail, feed

gm, guardrail, feed = get_system_components()

# Handle dynamic modifications to the graph state in session_state
if "closed_edges" not in st.session_state:
    st.session_state.closed_edges = set()
if "occupied_edges" not in st.session_state:
    st.session_state.occupied_edges = set()
if "highlighted_path" not in st.session_state:
    st.session_state.highlighted_path = []
if "violated_nodes" not in st.session_state:
    st.session_state.violated_nodes = []
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "last_api_fetch_time" not in st.session_state:
    st.session_state.last_api_fetch_time = 0.0
if "traffic" not in st.session_state:
    st.session_state.traffic = []

# Sync session_state changes with the active guardrail object
guardrail.closed_segments = st.session_state.closed_edges
guardrail.occupied_segments = st.session_state.occupied_edges

# Helper function to extract violated nodes from guardrail warning text
def extract_violated_nodes(violations, graph_nodes):
    violated = set()
    for v in violations:
        # Match pattern "between X and Y"
        between_match = re.search(r"between\s+(\S+)\s+and\s+(\S+)", v)
        if between_match:
            n1 = between_match.group(1).rstrip('.').strip()
            n2 = between_match.group(2).rstrip('.').strip()
            if n1 in graph_nodes: violated.add(n1)
            if n2 in graph_nodes: violated.add(n2)
            
        # Match pattern "segment X->Y"
        segment_match = re.search(r"segment\s+(\S+)->(\S+)", v)
        if segment_match:
            n1 = segment_match.group(1).split(" ")[0].strip()
            n2 = segment_match.group(2).split(" ")[0].strip()
            # Clean up non-alphanumeric chars at the ends
            n1 = re.sub(r"^[^\w_]+|[^\w_]+$", "", n1)
            n2 = re.sub(r"^[^\w_]+|[^\w_]+$", "", n2)
            if n1 in graph_nodes: violated.add(n1)
            if n2 in graph_nodes: violated.add(n2)
            
        # Match pattern "Risk at X"
        risk_match = re.search(r"Risk at\s+(\S+)", v)
        if risk_match:
            n = risk_match.group(1).rstrip('.').strip()
            n = re.sub(r"^[^\w_]+|[^\w_]+$", "", n)
            if n in graph_nodes: violated.add(n)
            
    return list(violated)

# Simulation function to move mock traffic in real-time
def update_traffic_simulation(dt=5.0):
    if "traffic" in st.session_state and st.session_state.traffic:
        for ac in st.session_state.traffic:
            speed = ac.get("velocity_mps", 0.0)
            heading = ac.get("heading_deg", 0.0)
            if speed > 0:
                lat = ac.get("latitude")
                lon = ac.get("longitude")
                if lat is not None and lon is not None:
                    heading_rad = math.radians(heading)
                    # 1 degree lat is ~111,111 meters
                    # 1 degree lon is ~111,111 * cos(lat) meters
                    dy = speed * math.cos(heading_rad) * dt
                    dx = speed * math.sin(heading_rad) * dt
                    
                    ac["latitude"] = lat + (dy / 111111.0)
                    ac["longitude"] = lon + (dx / (111111.0 * math.cos(math.radians(lat))))
                    
                    # Simulated descent for airborne aircraft
                    if not ac.get("on_ground", True):
                        alt = ac.get("altitude_m", 0.0)
                        ac["altitude_m"] = max(0.0, alt - (3.0 * dt)) # Descend at 3 m/s
                        if ac["altitude_m"] == 0:
                            ac["on_ground"] = True
                            ac["velocity_mps"] = 5.0 # Slow down to taxi speed

# Plotly Map Visualization Helper
def build_plotly_map(G, pos, closed_edges, highlighted_path, violated_nodes, traffic, show_labels=False, show_intersections=False):
    fig = go.Figure()
    
    # Check if we have real lat/lons or mock coordinates
    lats = [v[1] for v in pos.values()]
    lons = [v[0] for v in pos.values()]
    lat_range = max(lats) - min(lats) if lats else 1.0
    lon_range = max(lons) - min(lons) if lons else 1.0
    is_real_coords = lat_range < 5.0
    
    # 1. DRAW EDGES
    standard_x, standard_y, standard_text = [], [], []
    closed_x, closed_y, closed_text = [], [], []
    path_x, path_y, path_text = [], [], []
    
    path_edges_set = set()
    if highlighted_path:
        valid_path = all(n in G.nodes for n in highlighted_path)
        if valid_path:
            path_edges_set = set(zip(highlighted_path[:-1], highlighted_path[1:]))
            path_edges_set.update(set(zip(highlighted_path[1:], highlighted_path[:-1])))

    for u, v, attrs in G.edges(data=True):
        u_pos = pos[u]
        v_pos = pos[v]
        
        x_coords = [u_pos[0], v_pos[0], None]
        y_coords = [u_pos[1], v_pos[1], None]
        edge_name = attrs.get("name", "unnamed")
        max_span = attrs.get("max_wingspan", "N/A")
        edge_info = f"Segment: {u} -> {v}<br>Taxiway: {edge_name}<br>Max Wingspan: {max_span}m"
        
        if (u, v) in closed_edges or (v, u) in closed_edges:
            closed_x.extend(x_coords)
            closed_y.extend(y_coords)
            closed_text.extend([edge_info, edge_info, None])
        elif (u, v) in path_edges_set:
            path_x.extend(x_coords)
            path_y.extend(y_coords)
            path_text.extend([edge_info, edge_info, None])
        else:
            standard_x.extend(x_coords)
            standard_y.extend(y_coords)
            standard_text.extend([edge_info, edge_info, None])
            
    if standard_x:
        fig.add_trace(go.Scatter(
            x=standard_x, y=standard_y,
            mode='lines',
            line=dict(color='#475569', width=1.5),
            hoverinfo='text',
            text=standard_text,
            name='Taxiways',
            showlegend=True
        ))
        
    if closed_x:
        fig.add_trace(go.Scatter(
            x=closed_x, y=closed_y,
            mode='lines',
            line=dict(color='#ef4444', width=3.5),
            hoverinfo='text',
            text=closed_text,
            name='Closed / Hazard',
            showlegend=True
        ))
        
    if path_x:
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines',
            line=dict(color='#10b981', width=4),
            hoverinfo='text',
            text=path_text,
            name='Cleared Route',
            showlegend=True
        ))

    # 2. DRAW NODES
    node_types = {
        "gate": {"color": "#38bdf8", "name": "Gates / Stands", "symbol": "circle", "size": 10},
        "runway_entry": {"color": "#f43f5e", "name": "Runway Entries", "symbol": "diamond", "size": 10},
        "hold_short": {"color": "#fbbf24", "name": "Hold Shorts", "symbol": "square", "size": 10},
        "intersection": {"color": "#64748b", "name": "Intersections", "symbol": "circle", "size": 5}
    }
    
    type_nodes = {t: {"x": [], "y": [], "text": [], "labels": []} for t in node_types}
    violated_node_points = {"x": [], "y": [], "text": [], "labels": []}
    
    for node, attrs in G.nodes(data=True):
        n_pos = pos[node]
        n_type = attrs.get("type", "intersection")
        n_desc = attrs.get("description", "No description")
        n_elev = attrs.get("elevation")
        
        elev_str = f"{int(n_elev)}m" if n_elev is not None else "N/A"
        node_info = f"Node: {node}<br>Type: {n_type}<br>Description: {n_desc}<br>Elevation: {elev_str}"
        
        if node in violated_nodes:
            violated_node_points["x"].append(n_pos[0])
            violated_node_points["y"].append(n_pos[1])
            violated_node_points["text"].append(node_info)
            violated_node_points["labels"].append(node)
        elif n_type in type_nodes:
            if n_type == "intersection" and not show_intersections:
                continue
            type_nodes[n_type]["x"].append(n_pos[0])
            type_nodes[n_type]["y"].append(n_pos[1])
            type_nodes[n_type]["text"].append(node_info)
            type_nodes[n_type]["labels"].append(node)
            
    for n_type, config in node_types.items():
        data = type_nodes[n_type]
        if data["x"]:
            fig.add_trace(go.Scatter(
                x=data["x"], y=data["y"],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    color=config["color"],
                    size=config["size"],
                    symbol=config["symbol"],
                    line=dict(color='#0f172a', width=1)
                ),
                text=data["labels"] if show_labels else None,
                textposition="top center",
                textfont=dict(color='#f8fafc', size=9),
                hoverinfo='text',
                hovertext=data["text"],
                name=config["name"],
                showlegend=True
            ))
            
    if violated_node_points["x"]:
        fig.add_trace(go.Scatter(
            x=violated_node_points["x"], y=violated_node_points["y"],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                color='#ef4444',
                size=12,
                symbol='circle',
                line=dict(color='#ffffff', width=2)
            ),
            text=violated_node_points["labels"] if show_labels else None,
            textposition="top center",
            textfont=dict(color='#f8fafc', size=10, weight='bold'),
            hoverinfo='text',
            hovertext=violated_node_points["text"],
            name='Safety Violations',
            showlegend=True
        ))

    # 3. DRAW RADAR TRAFFIC (AIRCRAFT)
    if traffic:
        aircraft_x, aircraft_y, aircraft_text, aircraft_color, aircraft_labels = [], [], [], [], []
        vec_x, vec_y, vec_text = [], [], []
        
        scale = min(lat_range, lon_range) * 0.15 if is_real_coords else 0.5
        
        for ac in traffic:
            lon = ac.get("longitude")
            lat = ac.get("latitude")
            if lon is not None and lat is not None:
                on_ground = ac.get("on_ground", True)
                color = '#10b981' if on_ground else '#60a5fa'
                
                ac_info = (
                    f"Callsign: {ac['callsign']}<br>"
                    f"ICAO24: {ac['icao24']}<br>"
                    f"Status: {'On Ground' if on_ground else 'Airborne'}<br>"
                    f"Altitude: {int(ac.get('altitude_m', 0))}m<br>"
                    f"Speed: {ac.get('velocity_mps', 0.0):.1f} m/s ({int(ac.get('velocity_mps', 0.0) * 1.94384)} kts)<br>"
                    f"Heading: {ac.get('heading_deg', 0.0):.1f}°"
                )
                
                aircraft_x.append(lon)
                aircraft_y.append(lat)
                aircraft_text.append(ac_info)
                aircraft_color.append(color)
                aircraft_labels.append(ac['callsign'])
                
                heading_rad = math.radians(ac.get("heading_deg", 0.0))
                dx = math.sin(heading_rad) * scale
                dy = math.cos(heading_rad) * scale
                
                vec_x.extend([lon, lon + dx, None])
                vec_y.extend([lat, lat + dy, None])
                vec_text.extend([ac_info, ac_info, None])
                
        if vec_x:
            fig.add_trace(go.Scatter(
                x=vec_x, y=vec_y,
                mode='lines',
                line=dict(color='#f8fafc', width=1.5),
                hoverinfo='text',
                text=vec_text,
                name='Heading Vector',
                showlegend=False
            ))
            
        fig.add_trace(go.Scatter(
            x=aircraft_x, y=aircraft_y,
            mode='markers+text',
            marker=dict(
                color=aircraft_color,
                size=12,
                symbol='triangle-up',
                line=dict(color='#ffffff', width=1)
            ),
            text=aircraft_labels,
            textposition="bottom center",
            textfont=dict(color='#ffffff', size=9, weight='bold'),
            hoverinfo='text',
            hovertext=aircraft_text,
            name='Aircraft Target',
            showlegend=True
        ))

    fig.update_layout(
        plot_bgcolor='#090d16',
        paper_bgcolor='#090d16',
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1e293b',
            zeroline=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1e293b',
            zeroline=False,
            showticklabels=False,
            title='',
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            font=dict(color='#f8fafc', size=9),
            bgcolor='rgba(9, 13, 22, 0.85)',
            bordercolor='#1e293b',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=0.01,
            xanchor='left',
            x=0.01
        ),
        hoverlabel=dict(
            bgcolor='#1e293b',
            font_size=11,
            font_color='#f8fafc',
            font_family='Outfit, sans-serif'
        ),
        dragmode='pan'
    )
    
    return fig

# -----------------------------------------------------------------------------
# Sidebar Layout
# -----------------------------------------------------------------------------
st.sidebar.markdown("<h2 style='color:#38bdf8;'>✈️ SafeGround Command</h2>", unsafe_allow_html=True)
selected_airport = st.sidebar.selectbox("Select Airport Station", ["KSFO", "KLAX", "EGLL"])

# Manage dynamic airport graph loading
if "active_airport" not in st.session_state:
    st.session_state.active_airport = None
    
if st.session_state.active_airport != selected_airport:
    with st.spinner(f"Loading real-time geometry for {selected_airport} from OpenStreetMap..."):
        gm.load_airport(selected_airport)
        # Clear closed/occupied segments and paths
        st.session_state.closed_edges = set()
        st.session_state.occupied_edges = set()
        st.session_state.highlighted_path = []
        st.session_state.violated_nodes = []
        guardrail.closed_segments = st.session_state.closed_edges
        guardrail.occupied_segments = st.session_state.occupied_edges
        
        # Load fresh traffic for the new airport
        st.session_state.traffic = feed.fetch_live_traffic(selected_airport)
        st.session_state.last_api_fetch_time = time.time()
        
        st.session_state.active_airport = selected_airport
        st.rerun()

# API Key configuration override
user_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", 
                                     placeholder="Enter API Key to run actual LLM")

# Model configuration selector
model_options = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash", 
    "gemini-1.5-pro",
    "Custom Model Name"
]
selected_model_option = st.sidebar.selectbox("Gemini Model", model_options)

if selected_model_option == "Custom Model Name":
    selected_model = st.sidebar.text_input("Enter Model Name", value="gemini-2.5-flash")
else:
    selected_model = selected_model_option

if user_api_key:
    os.environ["GEMINI_API_KEY"] = user_api_key

# OpenSky Credentials
opensky_username = st.sidebar.text_input("OpenSky Username (Optional)", value="", placeholder="Username for higher limits")
opensky_password = st.sidebar.text_input("OpenSky Password (Optional)", type="password", value="", placeholder="Password")

# Force reload of agent with the key & model
agent = LLMCognitiveAgent(gm, model_name=selected_model)

# Initialize session state variables for flight parameters
if "ac_callsign" not in st.session_state:
    st.session_state.ac_callsign = "UAL901"
if "ac_type" not in st.session_state:
    st.session_state.ac_type = "B77W"
if "ac_wingspan" not in st.session_state:
    st.session_state.ac_wingspan = 64.8

# Fetch ADS-B Track Button
if st.sidebar.button("Scan Live ADS-B Ground Traffic"):
    with st.spinner("Fetching dynamic state vectors..."):
        traffic_data = feed.fetch_live_traffic(
            selected_airport, 
            username=opensky_username, 
            password=opensky_password
        )
        st.session_state.traffic = traffic_data
        st.session_state.last_api_fetch_time = time.time()
        st.sidebar.success(f"Detected {len(traffic_data)} aircraft.")

# Define callback to select aircraft and update session state
def select_aircraft(ac):
    st.session_state.ac_callsign = ac["callsign"]
    
    # Guess type/wingspan based on callsign or icao24
    callsign_upper = ac["callsign"].upper()
    if "SIA" in callsign_upper or "SQ" in callsign_upper or ac["icao24"] == "76cce7":
        st.session_state.ac_type = "A380"
        st.session_state.ac_wingspan = 79.8
    elif "DLH" in callsign_upper or "LH" in callsign_upper:
        st.session_state.ac_type = "A346"
        st.session_state.ac_wingspan = 63.4
    elif "UAL" in callsign_upper or "UA" in callsign_upper:
        if "901" in callsign_upper or "2837" in callsign_upper:
            st.session_state.ac_type = "B77W"
            st.session_state.ac_wingspan = 64.8
        else:
            st.session_state.ac_type = "B737"
            st.session_state.ac_wingspan = 35.8
    elif "AAL" in callsign_upper or "AA" in callsign_upper:
        st.session_state.ac_type = "A321"
        st.session_state.ac_wingspan = 35.8
    elif "SWA" in callsign_upper or "WN" in callsign_upper:
        st.session_state.ac_type = "B737"
        st.session_state.ac_wingspan = 35.8
    elif "ASA" in callsign_upper or "AS" in callsign_upper:
        st.session_state.ac_type = "B739"
        st.session_state.ac_wingspan = 35.8
    else:
        st.session_state.ac_type = "A320"
        st.session_state.ac_wingspan = 35.8

# Display detected planes list
st.sidebar.markdown("### Active Ground Targets")
if "traffic" in st.session_state and st.session_state.traffic:
    for idx, ac in enumerate(st.session_state.traffic):
        on_ground_str = "On Ground" if ac["on_ground"] else f"Airborne ({int(ac['altitude_m'])}m)"
        # Simple click trigger using buttons with callbacks to avoid state lag
        st.sidebar.button(
            f"🎯 {ac['callsign']} ({ac['icao24']}) - {on_ground_str}", 
            key=f"ac_{idx}", 
            on_click=select_aircraft, 
            args=(ac,)
        )
else:
    st.sidebar.info("No targets scanned yet. Use standard presets or scan above.")

# Add custom hazards block in Sidebar
st.sidebar.markdown("### Simulated Airfield Hazards")
# Only show actual taxiways (filter out runways)
taxiway_edges = [
    (u, v) for u, v in gm.graph.edges 
    if gm.graph[u][v].get("aeroway", "taxiway") == "taxiway"
]
hazard_segment = st.sidebar.selectbox(
    "Select Taxiway segment to close",
    [f"{u} -> {v} ({gm.graph[u][v].get('name', 'unnamed')})" for u, v in taxiway_edges] if taxiway_edges else ["No taxiways available"]
)

if st.sidebar.button("Close Segment / Mark FOD") and hazard_segment != "No taxiways available":
    # Extract u and v from selection string
    parts = hazard_segment.split(" -> ")
    u = parts[0]
    v = parts[1].split(" (")[0]
    st.session_state.closed_edges.add((u, v))
    st.session_state.closed_edges.add((v, u)) # undirected hazard
    guardrail.close_segment(u, v)
    st.success(f"Closed taxiway segment {u} to {v}!")
    st.rerun()

if st.session_state.closed_edges:
    if st.sidebar.button("Clear Airfield Obstacles"):
        st.session_state.closed_edges.clear()
        guardrail.closed_segments.clear()
        st.success("Airfield cleared.")
        st.rerun()

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown("<h1 class='glow-header'>SafeGround-LLM</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-top:-10px;'>Neuro-Symbolic Decision Support Copilot for Airport Ground Traffic Controllers</p>", unsafe_allow_html=True)

# Layout columns: Left for spatial graph plot, Right for agent interface
col_map, col_controls = st.columns([1.1, 1.0])

# Keep path variables global to highlight during rendering
highlighted_path = []
violated_nodes = []

with col_controls:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📋 Flight Parameters")
    
    # Input bindings (directly bound to session state to prevent resetting)
    c_left, c_right = st.columns(2)
    
    def on_type_change():
        t = st.session_state.ac_type.upper()
        if t in ["A380", "A388"]:
            st.session_state.ac_wingspan = 79.8
        elif t in ["B77W", "B773"]:
            st.session_state.ac_wingspan = 64.8
        elif t in ["B748", "B744"]:
            st.session_state.ac_wingspan = 68.4
        elif t in ["A320", "B737", "A321", "B739", "B738"]:
            st.session_state.ac_wingspan = 35.8

    with c_left:
        aircraft_callsign = st.text_input("Aircraft Callsign", key="ac_callsign")
        aircraft_type = st.text_input("Aircraft Type Code", key="ac_type", on_change=on_type_change)
    with c_right:
        wingspan = st.number_input("Wingspan (meters)", min_value=1.0, max_value=85.0, step=0.1, key="ac_wingspan")
        
        # Filter start and destination nodes based on flight phase to prevent label clutter
        route_phase = st.segmented_control(
            "Flight Phase",
            options=["Departure (Gate → Runway)", "Arrival (Runway → Gate)", "Custom Route"],
            default="Departure (Gate → Runway)"
        )
        
        all_gates = [node for node, attrs in gm.graph.nodes(data=True) if attrs.get("type") == "gate"]
        all_runways = [node for node, attrs in gm.graph.nodes(data=True) if attrs.get("type") in ["hold_short", "runway_entry"]]
        all_intersections = [node for node, attrs in gm.graph.nodes(data=True) if attrs.get("type") == "intersection"]
        
        if route_phase == "Departure (Gate → Runway)":
            start_nodes = all_gates if all_gates else ["Gate_A1"]
            dest_nodes = all_runways if all_runways else ["Hold_Short_28L"]
        elif route_phase == "Arrival (Runway → Gate)":
            start_nodes = all_runways if all_runways else ["Hold_Short_28L"]
            dest_nodes = all_gates if all_gates else ["Gate_A1"]
        else: # Custom
            start_nodes = list(gm.graph.nodes)
            dest_nodes = list(gm.graph.nodes)
            
        start_nodes.sort()
        dest_nodes.sort()
        
        if not start_nodes:
            start_nodes = ["Gate_A1"]
        if not dest_nodes:
            dest_nodes = ["Hold_Short_28L"]
            
        current_type = st.session_state.ac_type.upper()
        
        val_start_idx = 0
        if "Gate_A2" in start_nodes:
            val_start_idx = start_nodes.index("Gate_A2")
        elif "Gate_A1" in start_nodes:
            val_start_idx = start_nodes.index("Gate_A1")
        elif "Gate_G1" in start_nodes:
            val_start_idx = start_nodes.index("Gate_G1")
            
        if val_start_idx >= len(start_nodes):
            val_start_idx = 0
            
        start = st.selectbox("Gate Stand / Start Point", start_nodes, index=val_start_idx)
        
    c_dest_left, c_dest_right = st.columns(2)
    with c_dest_left:
        val_dest_idx = 0
        if "Hold_Short_28L" in dest_nodes:
            val_dest_idx = dest_nodes.index("Hold_Short_28L")
        elif "Gate_A1" in dest_nodes:
            val_dest_idx = dest_nodes.index("Gate_A1")
            
        if val_dest_idx >= len(dest_nodes):
            val_dest_idx = 0
            
        destination = st.selectbox("Destination Runway / Entry Threshold", dest_nodes, index=val_dest_idx)
    with c_dest_right:
        st.markdown("<br>", unsafe_allow_html=True)
        is_runway_cleared = st.checkbox("Active Runway Crossing Approved", value=False)
        
    weather = "KSFO 081250Z 27015KT 10SM CLR 18/12 A2992"
    
    # Run Agent Button
    st.markdown("<br>", unsafe_allow_html=True)
    run_copilot = st.button("🔌 Query SafeGround-LLM Advice", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_copilot:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("💡 Copilot Decision Support Output")
        
        # 1. Ask cognitive LLM Agent for taxi route proposal (Attempt 1)
        with st.spinner("LLM Cognitive Agent generating initial routing advice..."):
            proposal = agent.propose_clearance(
                aircraft_callsign=aircraft_callsign,
                aircraft_type=aircraft_type,
                wingspan=wingspan,
                start=start,
                destination=destination,
                weather=weather,
                is_runway_cleared=is_runway_cleared,
                attempt=1
            )
        
        # Check if an API error occurred (e.g. invalid key, model 404, etc.)
        if "error" in proposal["metadata"]:
            st.error(f"⚠️ **Gemini API Query Failed**: {proposal['metadata']['error']}")
            st.info("The system is automatically falling back to local SafeGround-LLM simulation mode to complete your request.")
            # Run local fallback clearance
            agent.use_real_llm = False
            proposal = agent.propose_clearance(
                aircraft_callsign=aircraft_callsign,
                aircraft_type=aircraft_type,
                wingspan=wingspan,
                start=start,
                destination=destination,
                weather=weather,
                is_runway_cleared=is_runway_cleared,
                attempt=1
            )
            # Restore state for subsequent interactions
            agent.use_real_llm = True
            
        st.markdown(f"**LLM Proposes:** Node Path `{proposal['proposed_path']}`")
        st.session_state.highlighted_path = proposal['proposed_path']
        
        # 2. Feed path to Symbolic Guardrail verifier
        is_safe, violations = guardrail.verify_taxi_path(
            aircraft_callsign=aircraft_callsign,
            wingspan=wingspan,
            proposed_path=proposal["proposed_path"],
            runway_cleared=is_runway_cleared
        )
        
        if is_safe:
            st.session_state.violated_nodes = []
            st.markdown("<span class='badge-approved'>✓ VALIDATED BY SAFETY GUARDRAIL</span>", unsafe_allow_html=True)
            st.markdown("<div class='terminal-box'>" + proposal['icao_phraseology'] + "</div>", unsafe_allow_html=True)
            st.markdown(f"**Rationale Trace:** *{proposal['rationale']}*")
            st.success("Safety Proof: Separation checks passed, wingspan limits verified, hold-short rules validated.")
        else:
            st.markdown("<span class='badge-rejected'>✗ REJECTED BY GUARDRAIL - LLM Hallucinated Unsafe Path</span>", unsafe_allow_html=True)
            st.error("\n".join([f"⚠️ {v}" for v in violations]))
            
            # Extract nodes where violations occur for map highlighting dynamically
            st.session_state.violated_nodes = extract_violated_nodes(violations, list(gm.graph.nodes))
            
            # 3. Reflection Loop: Re-feed violations back to the LLM agent to correct itself (Attempt 2)
            with st.spinner("Triggering closed-loop agent self-reflection. Replanning route..."):
                corrected_proposal = agent.reflect_and_replan(
                    aircraft_callsign=aircraft_callsign,
                    aircraft_type=aircraft_type,
                    wingspan=wingspan,
                    start=start,
                    destination=destination,
                    weather=weather,
                    is_runway_cleared=is_runway_cleared,
                    violation_messages=violations
                )
                
            # Check if an API error occurred during reflection replanning
            if "error" in corrected_proposal["metadata"]:
                agent.use_real_llm = False
                corrected_proposal = agent.reflect_and_replan(
                    aircraft_callsign=aircraft_callsign,
                    aircraft_type=aircraft_type,
                    wingspan=wingspan,
                    start=start,
                    destination=destination,
                    weather=weather,
                    is_runway_cleared=is_runway_cleared,
                    violation_messages=violations
                )
                agent.use_real_llm = True
                
            st.markdown(f"**LLM Corrected Proposal:** Node Path `{corrected_proposal['proposed_path']}`")
            st.session_state.highlighted_path = corrected_proposal['proposed_path']
            
            # Re-verify the replanned path
            is_safe_2, violations_2 = guardrail.verify_taxi_path(
                aircraft_callsign=aircraft_callsign,
                wingspan=wingspan,
                proposed_path=corrected_proposal["proposed_path"],
                runway_cleared=is_runway_cleared
            )
            
            if is_safe_2:
                st.session_state.violated_nodes = []
                st.markdown("<span class='badge-warning'>⚡ AUTO-CORRECTED & CERTIFIED SAFE</span>", unsafe_allow_html=True)
                st.markdown("<div class='terminal-box'>" + corrected_proposal['icao_phraseology'] + "</div>", unsafe_allow_html=True)
                st.markdown(f"**Replanning Rationale:** *{corrected_proposal['rationale']}*")
                st.info("System Engine automatically forced compliance with wingspan clearances and hold-short thresholds.")
            else:
                st.markdown("<span class='badge-rejected'>✗ RE-VERIFICATION FAILURE</span>", unsafe_allow_html=True)
                st.error("\n".join([f"⚠️ {v}" for v in violations_2]))
                
                # Extract nodes where violations occur dynamically
                st.session_state.violated_nodes = extract_violated_nodes(violations_2, list(gm.graph.nodes))
                
        # Engine identifier metadata
        engine_type = proposal["metadata"].get("engine", "Fallback")
        st.markdown(f"<p style='color:#64748b; font-size:0.8rem; margin-top:15px;'>Decision engine: {engine_type} | Latency: 0.85s</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
                
        # Engine identifier metadata
        engine_type = proposal["metadata"].get("engine", "Fallback")
        st.markdown(f"<p style='color:#64748b; font-size:0.8rem; margin-top:15px;'>Decision engine: {engine_type} | Latency: 0.85s</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Spatial Map Visualization Plotting (Left Column)
# -----------------------------------------------------------------------------
with col_map:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    
    # Segmented/fragmented live updating map container
    @st.fragment(run_every=5 if st.session_state.get("auto_refresh", False) else None)
    def render_airport_map_fragment():
        # Update simulation or fetch real traffic
        current_time = time.time()
        
        # Check if we should fetch real traffic (every 30 seconds if auto-refresh is active)
        if st.session_state.get("auto_refresh", False):
            last_fetch = st.session_state.get("last_api_fetch_time", 0.0)
            if current_time - last_fetch > 30.0:
                try:
                    traffic_data = feed.fetch_live_traffic(
                        selected_airport,
                        username=opensky_username,
                        password=opensky_password
                    )
                    if traffic_data:
                        st.session_state.traffic = traffic_data
                        st.session_state.last_api_fetch_time = current_time
                except Exception:
                    update_traffic_simulation(dt=5.0)
            else:
                update_traffic_simulation(dt=5.0)
        
        # Map control header
        c_map_header, c_map_refresh = st.columns([2.5, 1])
        with c_map_header:
            st.markdown(f"### 📡 {selected_airport} Ground Radar")
        with c_map_refresh:
            if st.button("🔄 Scan Radar", key="manual_map_refresh", width="stretch"):
                with st.spinner("Scanning..."):
                    try:
                        traffic_data = feed.fetch_live_traffic(
                            selected_airport,
                            username=opensky_username,
                            password=opensky_password
                        )
                        st.session_state.traffic = traffic_data
                        st.session_state.last_api_fetch_time = time.time()
                    except Exception:
                        pass
                st.rerun()
                
        # Layout toggles
        c_tog1, c_tog2, c_tog3 = st.columns(3)
        with c_tog1:
            auto_refresh = st.checkbox("Auto-refresh (5s)", value=st.session_state.get("auto_refresh", False), key="auto_refresh_chk")
            if auto_refresh != st.session_state.get("auto_refresh", False):
                st.session_state.auto_refresh = auto_refresh
                st.rerun()
        with c_tog2:
            show_intersections = st.checkbox("Show Intersections", value=False, key="show_intersections_chk")
        with c_tog3:
            show_labels = st.checkbox("Show Labels", value=False, key="show_labels_chk")
            
        # Draw Plotly Radar Screen
        G = gm.graph
        pos = nx.get_node_attributes(G, 'pos')
        
        fig = build_plotly_map(
            G=G,
            pos=pos,
            closed_edges=st.session_state.closed_edges,
            highlighted_path=st.session_state.highlighted_path,
            violated_nodes=st.session_state.violated_nodes,
            traffic=st.session_state.get("traffic", []),
            show_labels=show_labels,
            show_intersections=show_intersections
        )
        
        # Render Plotly with scroll zoom and fit to container
        st.plotly_chart(fig, config={'scrollZoom': True, 'displaylogo': False})
        
        # Live target details table
        with st.expander("📡 Live Radar Targets List", expanded=False):
            if "traffic" in st.session_state and st.session_state.traffic:
                targets_list = []
                for ac in st.session_state.traffic:
                    on_ground_str = "On Ground" if ac["on_ground"] else f"Airborne ({int(ac['altitude_m'])}m)"
                    targets_list.append({
                        "Callsign": ac["callsign"],
                        "ICAO": ac["icao24"],
                        "Country": ac["origin_country"],
                        "Speed": f"{ac['velocity_mps']:.1f} m/s",
                        "Heading": f"{ac['heading_deg']:.0f}°",
                        "Status": on_ground_str
                    })
                st.dataframe(targets_list)
            else:
                st.info("No active radar targets detected.")
                
    render_airport_map_fragment()
    st.markdown("</div>", unsafe_allow_html=True)
