import numpy as np
import datetime

class Agent:
    def __init__(self, bca):
        self.bca = bca
        self.zn0_temp = None
        self.zn1_temp = None
        self.zn2_temp = None
        self.fan_electric_power = None
        self.fan_mass_flow = None
        self.CO2_indoor_zn0 = None
        self.CO2_indoor_zn2 = None
        self.Occupancy_schedule = None
        self.state_size = (10, 1)
        self.action_size = 10
        self.state = None
        self.time_of_day = None

    def observation_function(self):
        self.time = self.bca.get_ems_data(['t_datetimes'])
        if self.time < datetime.datetime.now():
            var_data = self.bca.get_ems_data(list(self.bca.tc_var.keys()))
            weather_data = self.bca.get_ems_data(list(self.bca.tc_weather.keys()), return_dict=True)
            self.zn0_temp = var_data[0]
            self.zn1_temp = var_data[1]
            self.zn2_temp = var_data[2]
            self.fan_electric_power = var_data[3]
            self.fan_mass_flow = var_data[4]
            self.CO2_in_zn0_con = var_data[5]
            self.CO2_in_zn2_con = var_data[6]
            self.Occupancy_schedule = var_data[7]
            self.state = self.get_state(var_data, weather_data)

    def actuation_function(self):
        action = self.act(self.state)
        fan_flow_rate = action * (1.35 / 10)
        current_temp = self.state[1] * 18 + 35
        if current_temp > 35:
            fan_flow_rate = 1.35
        return {'air_loop_fan_mass_flow_actuator': fan_flow_rate}

    def get_state(self, var_data, weather_data):
        self.time_of_day = self.bca.get_ems_data(['t_hours'])
        weather_data = list(weather_data.values())[:2]
        state = np.concatenate((np.array([self.time_of_day]), var_data, weather_data))
        state[0] = state[0] / 24
        state[1] = (state[1] - 35) / 18
        state[2] = (state[2] - 35) / 18
        state[3] = (state[3] - 35) / 18
        state[4] = state[4] / 467.63
        state[5] = state[5] / 1.35
        state[6] = state[6] / 1000
        state[7] = state[7] / 1000
        state[8] = state[8]
        return state

    def act(self, state):
        co2_indoor_con = state[6]
        if co2_indoor_con < 0.7:
            action = 10
        elif 0.7 <= co2_indoor_con < 0.9:
            action = 10
        else:
            action = 10
        return action
