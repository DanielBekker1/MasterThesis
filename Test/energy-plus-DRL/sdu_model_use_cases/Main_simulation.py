import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from emspy import BcaEnv, EmsPy
from agent import Agent  # Import of my Agent class from a separate script

#might need the following lines:
# from pyenergyplus import api #Importing from folder, therefore a warning may show
# from pyenergyplus.api import EnergyPlusAPI

# File paths and setup
ep_path = 'C:\\EnergyPlusV24-1-0'
idf_file_name = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\ReMoni_OS_Model_high_heat.idf"
ep_weather_path = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\DNK_MJ_Aarhus.AP.060700_TMYx.2007-2021.epw"
cvs_output_path = r'C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\dataframe_output.csv'

# State space variables
zn0 = 'Thermal Zone: Software_Office_1'
zn1 = 'Thermal Zone: Finance_Office_1'
zn2 = 'Thermal Zone: Hardware_Corridor'

tc_vars = {
    'zn_soft1_temp': ('Zone Air Temperature', zn0),
    'zn_finance1_temp': ('Zone Air Temperature', zn1),
    'zn_hardware_corri_temp': ('Zone Air Temperature', zn2),
    'air_loop_fan_electric_power': ('Fan Electricity Rate', 'Const Spd Fan'),
    'air_loop_fan_mass_flow': ('Fan Air Mass Flow Rate', 'Const Spd Fan'),
    'Indoor_CO2_zn0': ('Zone Air CO2 Concentration', zn0),
    'Indoor_CO2_zn2': ('Zone Air CO2 Concentration', zn2),
    'Occupancy_schedule': ('Schedule Value', 'Office Occupancy')
}

tc_weather = {
    'oa_rh': ('outdoor_relative_humidity'),
    'oa_db': ('outdoor_dry_bulb'),
    'oa_pa': ('outdoor_barometric_pressure'),
    'sun_up': ('sun_is_up'),
    'rain': ('is_raining'),
    'snow': ('is_snowing'),
    'wind_dir': ('wind_direction'),
    'wind_speed': ('wind_speed')
}

tc_actuators = {
    'air_loop_fan_mass_flow_actuator': ('Fan', 'Fan Air Mass Flow Rate', 'CONST SPD FAN')
}

calling_point_for_callback_fxn = EmsPy.available_calling_points[7]
sim_timesteps = 6

# Create Building Energy Simulation Instance
sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name,
    timesteps=sim_timesteps,
    tc_vars=tc_vars,
    tc_intvars={},
    tc_meters={},
    tc_actuator=tc_actuators,
    tc_weather=tc_weather
)

# Create agent instance
my_agent = Agent(sim)

# Set callback functions for observation and actuation
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,
    observation_function=my_agent.observation_function,
    actuation_function=my_agent.actuation_function,
    update_state=True,
    update_observation_frequency=1,
    update_actuation_frequency=1
)

# Run the simulation
sim.run_env(ep_weather_path)
sim.reset_state()

# Sample output data
output_dfs = sim.get_df(to_csv_file=cvs_output_path)

# Plot results
df_var = output_dfs['var']
Start_period = 0
num_data_points = Start_period + (24 * 6 * 31)
week_data = df_var.iloc[Start_period:num_data_points]

fig, (ax1, ax2, ax5) = plt.subplots(ncols=3, figsize=(12, 12))
week_data.plot(y='zn_soft1_temp', ax=ax1, color='red')
week_data.plot(y='zn_finance1_temp', ax=ax2, color='red')
week_data.plot(y='zn_hardware_corri_temp', ax=ax5, color='red')

ax1.set_title('zn_soft1_temp')
ax2.set_title('zn_finance1_temp')
ax1.set_xlabel('Time')
ax2.set_xlabel('Time')

fig, (ax3, ax4) = plt.subplots(ncols=2, figsize=(12, 12))
week_data.plot(y='air_loop_fan_mass_flow', ax=ax3, color='green')
week_data.plot(y='air_loop_fan_electric_power', ax=ax4, color='blue')

ax3.set_ylabel('Fan Mass Flow Rate (kg/s)', color='green')
ax3.set_xlabel('Time')
ax4.set_ylabel('Fan Electricity power [W]', color='blue')
ax4.set_xlabel('Time')
ax3.legend(loc='upper left')
ax4.legend(loc='upper left')
