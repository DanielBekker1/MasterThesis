import sys
# sys.path.insert(0, r'C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\energy-plus-DRL\RL-EmsPy\emspy')
sys.path.insert(0, r'C:\EnergyPlusV24-1-0')
from pyenergyplus import api #Importing from folder, therefore a warning may show
from pyenergyplus.api import EnergyPlusAPI

import numpy as np
import datetime
from emspy import BcaEnv
from emspy import EmsPy
# from bca import BcaEnv
import datetime
import matplotlib.pyplot as plt

# -- FILE PATHS (Update as necessary) --
ep_path = r'C:\EnergyPlusV24-1-0'
idf_file_name_main = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\ReMoni_OS_Model_jan_day.idf"  # building energy model (BEM) IDF file
idf_file_name_secondary = r'C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\ReMoni_OS_Model_jan_second_sim.idf'
ep_weather_path = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\DNK_MJ_Aarhus_jan_2007-2021.epw" #  EPW weather file entire year

zn0 = 'Thermal Zone: Software_Office_1' #name of the zone to control 
zn1 = 'Thermal Zone: Finance_Office_1'
zn2 = 'Thermal Zone: Hardware_Corridor'

tc_intvars = {}
tc_vars = {
    # Building
    #'hvac_operation_sched': ('Schedule Value', 'HtgSetp 1'),  # is building 'open'/'close'?
    # -- Zone 0 (Core_Zn) --
    'zn_soft1_temp': ('Zone Air Temperature', zn0),
    'zn_finance1_temp': ('Zone Air Temperature', zn1),
    'zn_hardware_corri_temp': ('Zone Air Temperature', zn2),
    'air_loop_fan_electric_power': ('Fan Electricity Rate', 'Const Spd Fan'),    # Electricity usage of the fan in HVAC system 
    'air_loop_fan_mass_flow': ('Fan Air Mass Flow Rate', 'Const Spd Fan'),
    'Indoor_CO2_zn0' : ('Zone Air CO2 Concentration',zn0),  #Indoor CO2 concentration. affected by ventilation and infil.
    'Indoor_CO2_zn2' : ('Zone Air CO2 Concentration',zn2),  #Indoor CO2 concentration. affected by ventilation and infil.
    'Occupancy_schedule' : ('Schedule Value', 'Office Occupancy'), 
    # 'ventil_zn0' : ('Zone Ventilation Mass Flow Rate',zn0),
        # deg C
}
tc_meters = {} # empty, don't need any
tc_weather = {
    'oa_rh': ('outdoor_relative_humidity'),  # %RH
    'oa_db': ('outdoor_dry_bulb'),  # deg C
    'oa_pa': ('outdoor_barometric_pressure'),  # Pa
    'sun_up': ('sun_is_up'),  # T/F
    'rain': ('is_raining'),  # T/F
    'snow': ('is_snowing'),  # T/F
    'wind_dir': ('wind_direction'),  # deg
    'wind_speed': ('wind_speed')  # m/s
}

# ACTION SPACE
tc_actuators = {
    # HVAC Control Setpoints
    #'zn0_CO2_con': ('Zone Temperature Control', 'Cooling Setpoint', zn0),  # deg C
    'air_loop_fan_mass_flow_actuator' : ('Fan','Fan Air Mass Flow Rate', 'CONST SPD FAN'),  # kg/s
    # 'air_loop_Ventil_flow_rate_actuator' : ('Sizing:System', 'Main Supply Volume Flow Rate','REMONI_VENTILATION'),
}

calling_point_for_callback_fxn = EmsPy.available_calling_points[7]
sim_timesteps = 2  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)

# -- Create environment both simulations
# MAIN SIMULATION
main_sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name_main,
    timesteps=sim_timesteps,
    tc_vars=tc_vars, tc_intvars=tc_intvars, tc_meters=tc_meters, tc_actuator=tc_actuators, tc_weather=tc_weather
)

# SECONDARY SIMULATION 
secondary_sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name_secondary,
    timesteps=sim_timesteps,
    tc_vars=tc_vars, tc_intvars=tc_intvars, tc_meters=tc_meters, tc_actuator=tc_actuators, tc_weather=tc_weather
)

#Trigger Timestep
trigger_timestep = 10  # timestep to start the secondary simulation

# -- Main Agent Class
class MainAgent:
    def __init__(self, bca_main, bca_secondary):
        self.bca_main = bca_main
        self.bca_secondary = bca_secondary
        self.trigger_timestep = trigger_timestep
        self.timestep = 0  # Track timestep in main simulation

    def observation_function(self):
        # Fetch main simulation time
        self.timestep += 1
        # self.timestep = self.bca_main.get_ems_data(['t_simulation_time'])  
        if self.timestep >= self.trigger_timestep:
            # Get data from simulation at current timestep (and calling point) using ToC names
            var_data = self.bca_main.get_ems_data(list(self.bca_main.tc_var.keys()))
            weather_data = self.bca_main.get_ems_data(list(self.bca_main.tc_weather.keys()), return_dict=True)

            self.time_of_day = self.bca_main.get_ems_data(['t_hours'])


            # get specific values from MdpManager based on name
            self.zn0_temp = var_data[0]                             # Correct data collected that will be relevant for m chase
            self.zn1_temp = var_data[1]
            self.zn2_temp = var_data[2]
            self.fan_electric_power = var_data[3]               # W
            self.fan_mass_flow = var_data[4]                    # kg/s

            self.CO2_in_zn0_con = var_data[5]                      # ppm
            self.CO2_in_zn2_con = var_data[6]                      # ppm

            self.Occupancy_schedule  = var_data[7]                   # ppm
            # self.ventil_zn0 = var_data[8]
            
            print(f"Main simulation at timestep {self.timestep}. Data: {var_data}")
            
            self.state = self.get_state(var_data,weather_data)
        # Check if it's time to start the secondary simulation
        if self.timestep >= self.trigger_timestep:
            print(f"Main simulation reached timestep {self.trigger_timestep}. Starting secondary simulation.")
            self.start_secondary_simulation()

    def start_secondary_simulation(self):
        """
        Starts the secondary simulation at the specified timestep (trigger_timestep) in the main simulation.
        """
        # Running secondary simulation
        self.bca_secondary.run_env(ep_weather_path)

        # Get results from secondary simulation (if needed) and apply to main
        secondary_results = self.bca_secondary.get_df()
        print("Secondary simulation completed. Results:", secondary_results)

    def actuation_function(self):
        actuator_name = 'air_loop_fan_mass_flow_actuator'

        if self.timestep < datetime.datetime.now():
            action = self.act(self.state)
            fan_flow_rate = action*(1.35/10)


        current_temp = self.state[1] * 18 + 35
        if current_temp > 35:
            fan_flow_rate = 1.35

        return {actuator_name: fan_flow_rate,}
    def get_state(self, var_data, weather_data):

        #State:                  MAX:                  MIN:
        # 0: time of day        24                    0
        # 1: zone0_temp         35                    18
        # 2: zone1_temp         35                    18
        # 3: zone2_temp         35                    18
        
        # 4: fan_electric_power 77.94                 0         sum of an hour (467.63)
        # 5: fan_mass_flow      1.35                  0         sum of an hour (1.35) divided by 6 = 0.225
        
        # 6: CO2 con indoor     1000                  0
        # 7: infil software     1000                  0
        # 6: ppd                100                   0        
        # 7: outdoor_rh         100                   0  
        # 8: outdoor_temp       10                   -10

        time_of_day = self.bca_secondary.get_ems_data(['t_hours'])
        
        if isinstance(time_of_day, np.ndarray):
            time_of_day = time_of_day.flatten()[0]

        weather_data = list(weather_data.values())[:2]



        #concatenate self.time_of_day , var_data and weather_data
        state = np.concatenate((np.array([self.time_of_day]),var_data,weather_data)) 

        #normalize each value in the state according to the table above
        state[0] = state[0]/24
        state[1] = (state[1]-18)/18
        state[2] = (state[2]-18)/18
        state[3] = (state[3]-18)/18
        
        state[4] = state[4]/467.63
        state[5] = state[5]/1.35
        state[6] = state[6]/1000
        state[7] = state[7]/1000

        state[8] = (state[8]+10)/20
        # state[6] = state[6]/100

        # if len(weather_data) == 2:
        #     state[7] = state[7]/100
        #     state[8] = (state[8]+10)/20

        return state
        


# -- Initialize MainAgen
main_agent = MainAgent(main_sim, secondary_sim)

# -- Main Simulation with Callback to Start Secondary --
main_sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,  # Timestep calling point (adjust based on needs)
    observation_function=main_agent.observation_function,  # Check at each timestep if secondary sim should start
    actuation_function=main_agent.actuation_function,
    update_state=True,  # use this callback to update the EMS state
    update_observation_frequency=1,  # linked to observation update
    update_actuation_frequency=1
)

# -- RUN MAIN SIMULATION --
print("Starting main simulation...")
main_sim.run_env(ep_weather_path)
print("Main simulation completed.")