# SafeGround-LLM: Neuro-Symbolic LLM Agent for Airport Ground Control

This repository contains the prototype implementation and research skeleton for **SafeGround-LLM**, a hybrid neuro-symbolic decision-support system designed to assist airport ground traffic controllers.

For details on the research paper proposal, target journals, research questions, and evaluation metrics, please refer to:
👉 **[research_proposal.md](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/research_proposal.md)**

---

## 🚀 Key Features

1. **Neuro-Symbolic Architecture**: Couples neural LLM reasoning (for understanding context, SOPs, and speech) with a deterministic, rule-based symbolic engine to verify path safety.
2. **Spatio-Temporal Graph Grounding**: Represents airport maps as node-edge networks, allowing the LLM to query layouts using structural APIs rather than raw geographic coordinates (eliminating spatial reasoning errors).
3. **Guardrail-in-the-Loop Reflection**: If the LLM proposes an unsafe route (e.g., runway incursion or wingtip separation violation), the Symbolic Guardrail rejects the command and passes structured feedback back to the LLM for self-correction.
4. **ICAO Phraseology Synthesis**: Translates validated paths into compliant, standardized aviation radio instructions with human-readable rationale checks.

---

## 📁 Repository Structure

*   **[research_proposal.md](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/research_proposal.md)**: Academic research paper proposal and blueprint.
*   **[requirements.txt](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/requirements.txt)**: List of dependencies (`networkx`, `pydantic`, etc.).
*   **[src/](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/src)**: Source code folder.
    *   **[src/airport_graph.py](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/src/airport_graph.py)**: Loads and manages the topological map representing runways, gates, and taxiways using `networkx`.
    *   **[src/symbolic_guardrail.py](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/src/symbolic_guardrail.py)**: Deterministic checker enforcing separation minimums, closed taxiways, and runway occupancy.
    *   **[src/llm_agent.py](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/src/llm_agent.py)**: Simulates the cognitive LLM reasoning agent with tool-calling capabilities and self-reflection loops.
    *   **[src/main.py](file:///C:/Users/laxmi/OneDrive/Desktop/Paper_2/src/main.py)**: Runs live scenarios demonstrating the system detecting safety violations and automatically replanning.

---

## 💻 How to Run the Demo

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute the Simulation
Run the module using:
```bash
python -m src.main
```

---

## 📈 Demonstrated Scenarios

1.  **Scenario 1: Heavy Aircraft Wingtip Separation**:
    *   *Problem*: A Boeing 777-300ER (wingspan 64.8m) wants to taxi from Gate A1. The LLM initially routes it through Taxiway Alpha (limit 36m) and Taxiway Charlie (limit 52m) and tries to cross Runway 28L without crossing clearance.
    *   *Detection*: The guardrail catches the wingspan limits exceeded and the runway entry.
    *   *Correction*: The LLM replans starting from Gate A2, taxiing along Taxiway Delta (65m limit), and holds short of Runway 28L.
2.  **Scenario 2: Hold-Short Enforcement**:
    *   *Problem*: An Airbus A320 attempts to enter Runway 28L entry threshold without active controller crossing permission.
    *   *Detection*: The guardrail flags a Runway Incursion Risk.
    *   *Correction*: The LLM refines the clearance to hold short of Runway 28L.
