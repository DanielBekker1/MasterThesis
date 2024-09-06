import sys
sys.path.insert(0, r'C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\energy-plus-DRL\RL-EmsPy\emspy')
sys.path.insert(0, r'C:\EnergyPlusV24-1-0')


from pyenergyplus import api #Importing from folder, therefore a warning may show
from pyenergyplus.api import EnergyPlusAPI
import numpy as np
# from emspy import BcaEnv
from emspy import EmsPy
from bca import BcaEnv
import datetime
import matplotlib.pyplot as plt
import tkinter
from scipy.optimize import minimize


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
cvs_output_path = r'C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\dataframe_output.csv'

#model_path = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Test\energy-plus-DRL\sdu_model_use_cases\test.h5"

# model_path = load_model('actor_model.h5')

# STATE SPACE (& Auxiliary Simulation Data)

zn0 = 'Thermal Zone: Software_Office_1' #name of the zone to control 
zn1 = 'Thermal Zone: Finance_Office_1'
zn2 = 'Thermal Zone: Hardware_Corridor'

# zn2 = ''
tc_intvars = {}  # empty, don't need any

########### Fetching the variables that I will need. Find the names of the available variables in the .edd file.

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
# -- Simulation Params --
calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  # 6-16 valid for timestep loop during simulation
sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)

# -- Create Building Energy Simulation Instance --
sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name,
    timesteps=sim_timesteps,
    tc_vars=tc_vars,
    tc_intvars=tc_intvars,
    tc_meters=tc_meters,
    tc_actuator=tc_actuators,
    tc_weather=tc_weather
)


class Agent:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self, bca: BcaEnv):
        self.bca = bca
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


        if self.time < datetime.datetime.now():
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

            
          
# Define thresholds - Can find example in model_test.py.

    def actuation_function(self):

      
        # action = self.act(self.state)
        actuator_name = 'air_loop_fan_mass_flow_actuator'
        # actuator_name = 'air_loop_Ventil_flow_rate_actuator'

        #From the example in model_test.py created by Sebastian, the fan flow rate is found
        #with a density of 1.204 kg/m3. The max flow rate of the fan is fixed to 1.12
        #The max flow rate is 1.35 kg/s
        if self.time < datetime.datetime.now():
            action = self.act(self.state)
            fan_flow_rate = action*(1.35/10)


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


    def calculate_cost(self, predicted_states, control_vars):

        self.electricity_price = [
                        0.100, 0.200, 0.100, 0.400, -0.600, 0.100
                                    ]
        
        current_hour = int(self.time_of_day * 24) % 6
        current_price = self.electricity_price[current_hour]


        energy_cost = np.sum(predicted_states) * current_price
        control_penalty = np.sum(control_vars ** 2)

        total_cost = energy_cost + control_penalty

        # print(f"Predicted States: {predicted_states}")
        # print(f"Control Variables: {control_vars}")
        # print(f"Total Cost: {total_cost}\n")

        print(f"Energy Cost: {energy_cost}, Control Penalty: {control_penalty}, Total Cost: {total_cost}")
        print(f"Electricity Price: {current_price}, Predicted States: {predicted_states}, Control Variables: {control_vars}\n")


        return total_cost

    def act(self, state):
        prediction_horizon = 10

        initial_guess = np.ones(prediction_horizon) * 5 # Initial fan speed. ~ equal to half speed.

        def objective(control_vars):
            predicted_states = self.predict_future_states(state, control_vars)


            cost = self.calculate_cost(predicted_states, control_vars)
            return cost
        
        constraints = (
            {
                'type': 'ineq',  # inequality constraint
                'fun': lambda x: x  #  x >= 0 (fan speed should be non-negative)
            },
            {
                'type': 'ineq',  
                'fun': lambda x: 1.35 - x  # The constraint is x <= 1.35 (fan speed should be below or equal to max value)
            },
            {
                'type': 'ineq',
                'fun': lambda x: 1.35 if self.state[1] * 18 + 35 > 35 else x #Ensure Fan speed equal to 1.35 with temp >= 35
            }

        )
    
        bounds = [(0.00001, 1.35)] * prediction_horizon # Should ensure 
        
        solution = minimize(objective, initial_guess, constraints = constraints, bounds = bounds, method = 'SLSQP')

        optimal_fan_speed = solution.x[0]

        return optimal_fan_speed
    

    def predict_future_states(self, state, control_vars):
        predicted_states = []
        for control in control_vars:
            next_state = self.simulate_next_state(state, control)
            predicted_states.append(next_state)
            state = next_state
        return np.array(predicted_states)
        

    def cost(self, predicted_states, control_vars):

        cost = 0
        for i in range(len(predicted_states)):
            temp_deviation = max(0, predicted_states[i][1] - 25)  # penalize temp > 25°C
            co2_penalty = max(0, predicted_states[i][6] - 0.9)  # penalize -> CO2 > 800 ppm
            energy_cost = control_vars[i] ** 2  # penalize high fan speeds
            cost += temp_deviation*10 + co2_penalty*10 + energy_cost
        return cost
        # prediction = self.Actor(state)              # Could be this part that i missunderstand - Only have one part of self.actor
        # action = np.random.choice(self.action_size, p=np.squeeze(prediction)) 
        # return action
        # return 0


    def simulate_next_state(self, state, control):
        # Implement a state transition model
        # For example, a linear approximation of the dynamics
        next_state = state + control  # (simplified example)
        return next_state

#  --- Create agent instance ---

my_agent = Agent(sim)



# --- Set your callback function (observation and/or actuation) function for a given calling point ---
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,
    observation_function=my_agent.observation_function,  # optional function
    actuation_function= my_agent.actuation_function,  # optional function
    # actuator_names=['air_loop_fan_mass_flow_actuator'], # Ensure the correct actuator name is used
    update_state=True,  # use this callback to update the EMS state
    update_observation_frequency=1,  # linked to observation update
    update_actuation_frequency=1  # linked to actuation update
)

# -- RUN BUILDING SIMULATION --

sim.run_env(ep_weather_path)
sim.reset_state()  # reset when done


# -- Sample Output Data --
output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too, possibility for creating a CSV file (GB in size)

# -- Plot Results --

df_var = output_dfs['var']

Start_period = 0
num_data_points = Start_period + (24 * 6 * 31)

week_data = df_var.iloc[Start_period:num_data_points]


# fig, (ax1, ax2, ax5) = plt.subplots(ncols=3, figsize=(12, 12))  # Remember to change the ncols, number ax and figsize.

# week_data.plot(y='zn_soft1_temp', ax=ax1, color='red')
# week_data.plot(y='zn_finance1_temp', ax=ax2, color='red')
# week_data.plot(y='zn_hardware_corri_temp', ax=ax5, color='red')
# ax1.set_title('zn_soft1_temp')
# ax2.set_title('zn_finance1_temp')
# ax1.set_xlabel('Time')
# ax2.set_xlabel('Time')

fig, (ax3, ax4) = plt.subplots(ncols=2, figsize=(12, 12))

week_data.plot(y='air_loop_fan_mass_flow', ax=ax3, color='green')
week_data.plot(y='air_loop_fan_electric_power', ax=ax4, color='blue')

ax3.set_ylabel('Fan Mass Flow Rate (kg/s)', color='green')
ax3.set_xlabel('Time')
ax4.set_ylabel('Fan Electricity power [W]', color='blue')
ax4.set_xlabel('Time')


plt.show()
# Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes