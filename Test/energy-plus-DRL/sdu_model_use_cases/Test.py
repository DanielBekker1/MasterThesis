import sys

sys.path.insert(0, r'C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\energy-plus-DRL\RL-EmsPy\emspy')
sys.path.insert(0, r'C:\EnergyPlusV24-1-0')
import emspy
print("print the dir:", dir(emspy))

print("print the filename from emspy", emspy.__file__)  # Get the path to the package source

from pyenergyplus import api #Importing from folder, therefore a warning may show
from pyenergyplus.api import EnergyPlusAPI
import numpy as np
# from emspy import BcaEnv
from emspy import EmsPy
from bca import BcaEnv
import datetime
import matplotlib.pyplot as plt
import tkinter

# -- FILE PATHS --
# * E+ Download Path *
ep_path = r'C:\EnergyPlusV24-1-0'  # path to E+ on system
# IDF File / Modification Pathss
# idf_file_name = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\ReMoni_OS_Model_jan_2.idf"  # building energy model (BEM) IDF file January
idf_file_name = r"C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\ReMoni_OS_Model_jan_2.idf"  # building energy model (BEM) IDF file
# Weather Path
# ep_weather_path = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\DNK_MJ_Aarhus_jan_2007-2021.epw"  # EPW weather file january
ep_weather_path = r"C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\DNK_MJ_Aarhus_jan_2007-2021.epw" #  EPW weather file entire year
# Output .csv Path (optional)
cvs_output_path = r'C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\dataframe_output_test.csv'


# STATE SPACE (& Auxiliary Simulation Data)

zn0 = 'Thermal Zone: Software_Office_1' #name of the zone to control 
zn1 = 'Thermal Zone: Finance_Office_1'
zn2 = 'Thermal Zone: Hardware_Corridor'


Start_period = 0
num_data_points = Start_period + (24 * 6 * 31)


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
    # 'air_loop_fan_mass_flow_actuator' : ('Fan','Fan Air Mass Flow Rate', 'CONST SPD FAN'),  # kg/s
    'ReMoni_Ventilation_actuator' : ('Fan','Fan Air Mass Flow Rate', 'CONST SPD FAN'),  # kg/s

    # 'ReMoni_Ventilation_actuator': ('Outdoor Air System', 'Airflow Rate', 'ReMoni_Ventilation'),
    # 'air_loop_Ventil_flow_rate_actuator' : ('Sizing:System', 'Main Supply Volume Flow Rate','REMONI_VENTILATION'),
}
# -- Simulation Params --
calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  # 6-16 valid for timestep loop during simulation
sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)



class Agent:
    def __init__(self, bca: BcaEnv):
        self.bca = bca
        # Initialize any additional variables needed

        # Should be same varialbes as obersevation_function

        # simulation data state
        self.zn0_temp = None  # deg C                       # self nr. 0
        self.zn1_temp = None  # deg C                       # self nr. 1
        self.zn2_temp = None  # deg C                       # Self nr. 2
        self.fan_electric_power = None  # W                 # self nr. 3
        self.fan_mass_flow = None   #kg/s                   # self nr. 4
        
        self.CO2_indoor_zn0 = None #ppm                     # self nr. 5
        self.CO2_indoor_zn2 = None #ppm                     # self nr. 6
        self.Occupancy_schedule = None #ppm                 # self nr. 7
        # self.ventil_zn0 = None                              # self nr. 8

    #   self.zn2_temp = None  # deg C                       # self nr. 5
        self.state_size = (10,1)
        self.action_size = 10

 

        self.state = None
        self.time_of_day = None

    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.bca.get_ems_data(['t_datetimes'])
        #check that self.time is less than current time

        if isinstance(self.time, list) and len(self.time) >0:
            self.time = self.time[0]


        if isinstance(self.time, datetime.datetime) and self.time < datetime.datetime.now():
            # Get data from simulation at current timestep (and calling point) using ToC names
            var_data = self.bca.get_ems_data(list(self.bca.tc_var.keys()))
            weather_data = self.bca.get_ems_data(list(self.bca.tc_weather.keys()), return_dict=True)
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

            self.state = self.get_state(var_data,weather_data)

            print(f"Current simulation time: {self.time}")


    def actuation_function(self):

      
        # action = self.act(self.state)
        # actuator_name = 'air_loop_fan_mass_flow_actuator'
        actuator_name = 'ReMoni_Ventilation_actuator'

        #From the example in model_test.py created by Sebastian, the fan flow rate is found
        #with a density of 1.204 kg/m3. The max flow rate of the fan is fixed to 1.12
        #The max flow rate is 1.35 kg/s
        # fan_flow_rate = action*(1.35/10)

        if self.time < datetime.datetime.now():
            action = self.act(self.state)
            max_ventilation_rate = 1.35/10
            ventilation_flow_rate = action * max_ventilation_rate


        # The part "should" control the fan flow rate to be maximum if the indoor temperature in zno is above 35
        current_temp = self.state[1] * 18 + 35
        if current_temp > 35:
            ventilation_flow_rate = max_ventilation_rate


        return {actuator_name: ventilation_flow_rate}


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

        self.time_of_day = self.bca.get_ems_data(['t_hours'])
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

    def set_setpoints(self, setpoints):
        # Apply the optimized setpoints to the simulation actuators
        self.bca.tc_actuator['ReMoni_Ventilation_actuator'] = setpoints['ReMoni_Ventilation_actuator']
        # Apply other setpoints as needed

    def simulate_step(self):
        # Simulate a single timestep using the environment
        self.bca.run_step()
        results = self.bca.get_observations()
        return results


class MPCController:
    def __init__(self, agent, prediction_horizon, control_horizon):
        self.agent = agent
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon


    def optimize(self, current_state, weather_data, price_data):
        # Return a fixed setpoint for testing
        return {"ReMoni_Ventilation_actuator": 0.5}

    def generate_setpoints(self):
        # Generate possible setpoints for optimization
        setpoints = []
        # Add logic to generate setpoints
        return setpoints

    def simulate_setpoints(self, setpoints, weather_data, price_data):
        # Simulate the model with the given setpoints
        self.agent.set_setpoints(setpoints)
        results = self.agent.simulate_step()
        return results

    def calculate_cost(self, simulation_results):
        # Define the cost function
        cost = 0
        # Add logic to calculate cost from simulation results
        return cost
    

def main():
    # Initialize the environment and agent
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

    agent = Agent(sim)

    agent.observation_function()
    current_state = agent.state

    mpc = MPCController(agent, prediction_horizon=5, control_horizon=1)

    weather_data = get_weather_data(ep_weather_path)
    price_data = get_price_data()

    # Run the simulation with MPC
    for _ in range(num_data_points):  # num_data_points or another termination condition
        setpoints = mpc.optimize(current_state, weather_data, price_data)
        agent.set_setpoints(setpoints)
        agent.simulate_step()
        current_state = agent.sim.get_observations()

    # Process the output data
    output_dfs = sim.get_df(to_csv_file=cvs_output_path)


    # Plot results (as you already do)
    # plot_results(output_dfs)
    output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too, possibility for creating a CSV file (GB in size)
    df_var = output_dfs['var']



    week_data = df_var.iloc[Start_period:num_data_points]
    fig, (ax1, ax2, ax5) = plt.subplots(ncols=3, figsize=(12, 12))  # Remember to change the ncols, number ax and figsize.
    week_data.plot(y='zn_soft1_temp', ax=ax1, color='red')
    week_data.plot(y='zn_finance1_temp', ax=ax2, color='red')
    week_data.plot(y='zn_hardware_corri_temp', ax=ax5, color='red')

    # ax1, ax2,
    # output_dfs['var'].plot(y='zn_soft1_temp', use_index=True, ax=ax1)
    ax1.set_title('zn_soft1_temp')

    # output_dfs['var'].plot(y='zn_finance1_temp', use_index=True, ax=ax2)
    ax2.set_title('zn_finance1_temp')

    # Mass Flow Rate
    # output_dfs['var'].plot(y='air_loop_fan_mass_flow', use_index=True, ax=ax3, color='green')
    # ax3.set_title('Fan Mass Flow Rate')

    # # Electricity Consumption
    # output_dfs['var'].plot(y='air_loop_fan_electric_power', use_index=True, ax=ax4, color='blue')
    # ax4.set_title('Fan Electric Power')

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
    plt.show()

def get_weather_data(epw_file_path):
    # Function to extract relevant weather data for the MPC
    # This can be similar to the one in the initial script "no_aagent_baseline"
    return(ep_weather_path)

def get_price_data():
    # Function to retrieve or simulate energy price data
    pass

class CO2ControlEnv(BcaEnv):
    def observation_function(self):
        """
        Retrieve the CO2 concentration from the environment.
        """
        co2_concentration = self.get_ems_data('Indoor_CO2_zn0', return_dict=True)['Indoor_CO2_zn0']
        return {'CO2_Concentration': co2_concentration}

    def act(self, state):
        """
        Adjust the fan speed based on the CO2 concentration.
        """
        co2_concentration = state[6]* 1000
        max_flow_rate = 1.12
        min_flow_rate = 0.1


        # Example control logic
        if co2_concentration > 700:  # Threshold for high CO2 concentration
            ventilation_flow_rate = max_flow_rate  # Max fan speed
        elif 700 <= co2_concentration < 900:  # Threshold for medium CO2 concentration
            ventilation_flow_rate = min_flow_rate + (max_flow_rate - min_flow_rate) * (co2_concentration - 700) / 200  # Medium fan speed
        else:
            ventilation_flow_rate = min_flow_rate

        return {'ReMoni_Ventilation_actuator': ventilation_flow_rate}

    def set_calling_point_and_callback_function(self):
        """
        Set up the calling point and callbacks for observation and actuation functions.
        """
        self.set_calling_point_and_callback_function(
            calling_point='BeginTimestepBeforePredictor',  # Suitable calling point
            observation_function=self.observation_function,
            actuation_function=self.actuation_function,
            update_state=True,
            update_observation_frequency=1,
            update_actuation_frequency=1
        )

# Running the simulation with the new subclass
if __name__ == '__main__':


    # Initialize the CO2ControlEnv class with necessary parameters
    co2_control_env = CO2ControlEnv(
        ep_path,
        idf_file_name,
        timesteps=24*4,  # Example timestep count
        tc_vars={'Indoor_CO2_zn0': ('Zone Air CO2 Concentration', zn0)},
        tc_intvars={},
        tc_meters={},
        tc_actuator={'ReMoni_Ventilation_actuator': ('Fan Mass Flow Rate', 'CONST SPD FAN')},
        tc_weather={}
    )

    # Run the simulation
    co2_control_env.run_simulation(ep_weather_path, cvs_output_path)


