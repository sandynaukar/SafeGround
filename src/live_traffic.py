import json
import requests
from typing import List, Dict

class OpenSkyTrafficFeed:
    """
    Interface to fetch live aircraft states around specific airports
    using the OpenSky Network's public API.
    
    Includes automatic mock data generation as a fallback if the API is
    rate-limited or offline.
    """
    # Coordinates bounding boxes for major airports
    AIRPORT_BOUNDS = {
        "KSFO": {
            "lamin": 37.60,
            "lamax": 37.64,
            "lomin": -122.42,
            "lomax": -122.35,
            "name": "San Francisco International Airport"
        },
        "KLAX": {
            "lamin": 33.92,
            "lamax": 33.96,
            "lomin": -118.45,
            "lomax": -118.38,
            "name": "Los Angeles International Airport"
        },
        "EGLL": {
            "lamin": 51.45,
            "lamax": 51.49,
            "lomin": -0.50,
            "lomax": -0.40,
            "name": "London Heathrow Airport"
        }
    }

    def fetch_live_traffic(self, airport_code: str = "KSFO", username: str = None, password: str = None) -> List[Dict]:
        """
        Fetches live aircraft state vectors inside the expanded airport bounding box.
        Supports basic auth if username/password are provided.
        """
        if airport_code not in self.AIRPORT_BOUNDS:
            print(f"[TRAFFIC] Unknown airport code '{airport_code}'. Defaulting to KSFO.")
            airport_code = "KSFO"

        bounds = self.AIRPORT_BOUNDS[airport_code]
        # Expand bounds by 0.10 degrees in all directions to capture approaching aircraft
        lamin = bounds["lamin"] - 0.10
        lamax = bounds["lamax"] + 0.10
        lomin = bounds["lomin"] - 0.12
        lomax = bounds["lomax"] + 0.12

        url = "https://opensky-network.org/api/states/all"
        params = {
            "lamin": lamin,
            "lamax": lamax,
            "lomin": lomin,
            "lomax": lomax
        }

        print(f"[TRAFFIC] Ingesting live ADS-B tracks for {bounds['name']} ({airport_code})...")
        auth = (username, password) if username and password else None
        
        try:
            # Short timeout to ensure responsiveness
            response = requests.get(url, params=params, auth=auth, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                states = data.get("states")
                if states:
                    return self._parse_opensky_states(states)
                else:
                    print("[TRAFFIC] No active aircraft currently in the bounding box.")
                    return []
            elif response.status_code == 429:
                print("[TRAFFIC WARNING] OpenSky API rate limit hit. Generating fallback simulation data.")
                return self._generate_mock_traffic(airport_code)
            else:
                print(f"[TRAFFIC WARNING] API returned status {response.status_code}. Generating fallback simulation data.")
                return self._generate_mock_traffic(airport_code)
                
        except Exception as e:
            print(f"[TRAFFIC ERROR] Failed to connect to OpenSky API: {str(e)}")
            print("[TRAFFIC] Utilizing simulated live airport traffic...")
            return self._generate_mock_traffic(airport_code)

    def _parse_opensky_states(self, states: List) -> List[Dict]:
        """
        Parses raw OpenSky state arrays into a clean structured list of dicts.
        """
        parsed_aircraft = []
        for state in states:
            aircraft = {
                "icao24": state[0],
                "callsign": state[1].strip() if state[1] else "UNKNOWN",
                "origin_country": state[2],
                "longitude": state[5],
                "latitude": state[6],
                "altitude_m": state[7] if state[7] is not None else 0.0,
                "on_ground": bool(state[8]),
                "velocity_mps": state[9] if state[9] is not None else 0.0,
                "heading_deg": state[10] if state[10] is not None else 0.0,
                "squawk": state[14] if state[14] else "0000"
            }
            parsed_aircraft.append(aircraft)
        return parsed_aircraft

    def _generate_mock_traffic(self, airport_code: str) -> List[Dict]:
        """
        Generates realistic aircraft state vectors for the simulation fallback.
        """
        if airport_code == "KSFO":
            return [
                {
                    "icao24": "a823e1",
                    "callsign": "UAL901",
                    "origin_country": "United States",
                    "longitude": -122.385,
                    "latitude": 37.618,
                    "altitude_m": 0.0,
                    "on_ground": True,
                    "velocity_mps": 5.1, # Taxi speed ~10 knots
                    "heading_deg": 270.0,
                    "squawk": "1200"
                },
                {
                    "icao24": "a342b4",
                    "callsign": "AAL424",
                    "origin_country": "United States",
                    "longitude": -122.378,
                    "latitude": 37.612,
                    "altitude_m": 0.0,
                    "on_ground": True,
                    "velocity_mps": 0.0, # Stationary at gate
                    "heading_deg": 90.0,
                    "squawk": "2311"
                },
                {
                    "icao24": "ac4233",
                    "callsign": "SWA182",
                    "origin_country": "United States",
                    "longitude": -122.392,
                    "latitude": 37.625,
                    "altitude_m": 450.0, # Descending on final approach
                    "on_ground": False,
                    "velocity_mps": 72.0, # ~140 knots landing speed
                    "heading_deg": 284.0,
                    "squawk": "4412"
                }
            ]
        elif airport_code == "KLAX":
            return [
                {
                    "icao24": "a11223",
                    "callsign": "DLH456",
                    "origin_country": "Germany",
                    "longitude": -118.412,
                    "latitude": 33.942,
                    "altitude_m": 0.0,
                    "on_ground": True,
                    "velocity_mps": 4.5,
                    "heading_deg": 180.0,
                    "squawk": "3341"
                }
            ]
        return []

if __name__ == "__main__":
    feed = OpenSkyTrafficFeed()
    # Test KSFO
    traffic = feed.fetch_live_traffic("KSFO")
    print(f"\n--- Current Traffic Detected: {len(traffic)} aircraft ---")
    for ac in traffic:
        status = "On Ground" if ac["on_ground"] else f"Airborne ({int(ac['altitude_m'])}m)"
        print(f"Callsign: {ac['callsign']:<8} | ICAO: {ac['icao24']} | Speed: {ac['velocity_mps'] * 1.94384:.1f} kt | Status: {status:<15} | Lat/Lon: {ac['latitude']:.4f}, {ac['longitude']:.4f}")
