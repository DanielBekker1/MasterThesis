import numpy as np
import sys
import pandas as pd
from eppy import modeleditor
from eppy.modelededitor import IDF
from eppy.EPlusInterfaceFunctions import readidf
from eppy.EPlusInterfaceFunctions import parse_idd
sys.path.insert(0, r'C:\EnergyPlusV24-1-0')
from pyenergyplus.api import EnergyPlusAPI

class BuildingModel:
    def __init__(self, idf_file, weather_file):
        self.api = EnergyPlusAPI()
        self.runtime = self.api.runtime
        self.idf_file = idf_file
        self.weather_file = weather_file
        self.simulation_completed = False

    def _callback(self, state):
        self.simulation_completed = True

    def simulate(self, setpoints, weather_data, price_data):
        self.simulation_completed = False

        # Register the callback
        self.runtime.callback_end_zone_timestep_after_zone_reporting(self._callback)
        
        # Run the EnergyPlus simulation
        self.runtime.run_energyplus(['-r', '-w', self.weather_file, '-d', 'output', self.idf_file])

        # Wait for the simulation to complete
        while not self.simulation_completed:
            self.runtime.run_energyplus(['-r', '-w', self.weather_file, '-d', 'output', self.idf_file])
        
        results = self.read_results()  # Implement this method to extract relevant data
        return results

    def read_results(self):
        # Implement the logic to read simulation results
        return {}

class MPCController:
    def __init__(self, model, prediction_horizon, control_horizon):
        self.model = model
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

    def optimize(self, current_state, weather_data, price_data):
        best_setpoints = None
        best_cost = float('inf')

        for setpoints in self.generate_setpoints():
            results = self.model.simulate(setpoints, weather_data, price_data)
            cost = self.calculate_cost(results)
            if cost < best_cost:
                best_cost = cost
                best_setpoints = setpoints

        return best_setpoints

    def generate_setpoints(self):
        # Generate possible setpoints for optimization
        setpoints = []
        # Add your setpoint generation logic here
        return setpoints

    def calculate_cost(self, simulation_results):
        # Define the cost function
        cost = 0
        # Add your cost calculation logic here
        return cost

def main():
    idf_file = r"C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\ReMoni_OS_Model_jan.idf"
    weather_file = r"C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\DNK_MJ_Aarhus_jan_2007-2021.epw"
    model = BuildingModel(idf_file, weather_file)
    
    mpc = MPCController(model, prediction_horizon=5, control_horizon=1)

    current_state = None
    weather_data = get_weather_data(weather_file)  # Load initial weather data
    price_data = get_price_data()  # Load initial price data

    while simulation_running():
        weather_data = get_weather_data(weather_file)
        price_data = get_price_data()
        setpoints = mpc.optimize(current_state, weather_data, price_data)
        model.simulate(setpoints, weather_data, price_data)

def simulation_running():
    # Define your simulation termination logic
    global iteration
    iteration += 1
    return iteration < max_iterations

def get_weather_data(epw_file_path, start_time=0, end_time=5):
    # Read the EPW file using eppy
    with open(epw_file_path, 'r') as f:
        epw_data = f.readlines()

    # Extract relevant data for the prediction horizon
    weather_data = []
    for line in epw_data[8+start_time:8+end_time]:
        data = line.split(',')
        weather_data.append({
            "temperature": float(data[6]),  # Dry bulb temperature
            "humidity": float(data[8])      # Relative humidity
        })

    # Convert to pandas DataFrame for easier manipulation
    df_weather = pd.DataFrame(weather_data)

    return df_weather.to_dict(orient='list')

def get_price_data():
    # Example logic to retrieve price data
    price_data = [0.10, 0.12, 0.11, 0.09, 0.13]  # Example prices for 5 hours
    return price_data

if __name__ == '__main__':
    iteration = 0
    max_iterations = 10

    main()
