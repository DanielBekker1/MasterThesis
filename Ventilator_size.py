"""
This script calculate the flow rate of the ventilation system at ReMoni.
Data is collected by the "Ventilation Loft" censor at ReMoni. 
The following sources is used to calculate the flow rate of the HVAC system.
INSERT SOURCES
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
##################### Use the ReMoni censor "Ventilation Loft" and collect the apparent power of the ventilation system.

#csv_file_path = r"Room IndoorAir data\lagerhal\largerhal.csv"
# csv_file_path = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\exportData_Ventilation_system.csv"



# def plot_ventilation(csv_file_path):
#     ventilation_data = pd.read_csv(csv_file_path)   

   
#     apparent_power = ventilation_data[ventilation_data['Data type'] == 'Hourly apparent-power'].copy()
#     Hourly_Uncalibrated = ventilation_data[ventilation_data['Data type'] == 'Hourly Uncalibrated'].copy()
    

#     # Convert timestamps to datetime
#     apparent_power['Timestamp'] = pd.to_datetime(apparent_power['Timestamp'])
#     Hourly_Uncalibrated['Timestamp'] = pd.to_datetime(Hourly_Uncalibrated['Timestamp'])

#     print(apparent_power.head)
  

    
#     fig, ax1 = plt.subplots(figsize=(14, 8))

#     # Plot of Apparent power
#     color = 'tab:red'
#     ax1.set_xlabel('Timestamp')
#     ax1.set_ylabel('Apparent Power (VA)', color=color)
#     ax1.plot(apparent_power['Timestamp'], apparent_power['Value'], color=color, label='Apparent Power (VA)')
#     ax1.tick_params(axis='y', labelcolor=color)

#     # Add legends
#     ax1.legend(loc='upper left')
 
    
#     plt.savefig(os.path.join(os.path.dirname(csv_file_path), 'ventilation_consumption.png'))      #save the plot as a .png file


#     apparent_power = apparent_power.drop(['Data type','UnitTypeInput Id', 
#                                           'UnitTypeInput Name', 'Unit Input Custom Name', 
#                                           'Timestamp short (UTC)', 'Unit Id', 'Unit Name'], axis=1)
#     print(apparent_power.head())

#     apparent_power.to_csv('Apperent_Power.csv', index=False)                                      #Save the Apparent power into a csv file
    
#     # Checking for missing values.
#     df = apparent_power
#     for col in range(df.shape[1]): 
#       n_miss = df.iloc[:, col].isnull().sum()
    
#     print('Column {} has {} missing values.'.format(col, n_miss))


#     #find the average power consumption over the first 10 hours
#     ap_avg = apparent_power[0:9]
#     ap_avg = ap_avg['Value'].mean()
#     print('Average of the apperent power:',ap_avg)

#     #Calculation of the maximum flow rate of the fan:
#     Power_factory = 0.9                                 #EU standard (Need a source other than Muhyiddine)
#     fan_efficiency = 0.6                                #Need a source for the efficiency of the fan
#     conversion = 0.000471947                            #Will give the flow rate in m^3/s and not cubic feet per minute

#     real_power = Power_factory*ap_avg
#     max_flow_rate = (real_power/fan_efficiency)*conversion
#     print('The maximum flow rate of the fan is: {}'.format(max_flow_rate))
    
# plot_ventilation(csv_file_path)

filepath = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Software_CO2.csv"

software_co2 = pd.read_csv(filepath)

co2_concentration = software_co2[software_co2['Data type'] == 'Hourly co2-concentration'].copy()

print(co2_concentration)

co2_concentration['Timestamp'] = pd.to_datetime(co2_concentration['Timestamp'])

# Extract the hour and the day of the week from the timestamp
co2_concentration['Hour'] = co2_concentration['Timestamp'].dt.hour
co2_concentration['DayOfWeek'] = co2_concentration['Timestamp'].dt.dayofweek

# Classify days as either weekday (0-4) or weekend (5-6)
co2_concentration['DayType'] = co2_concentration['DayOfWeek'].apply(lambda x: 'Weekday' if x < 5 else 'Weekend')

# Group by hour and day type, then calculate the mean value
hourly_avg_co2 = co2_concentration.groupby(['DayType', 'Hour'])['Value'].mean().unstack(level=0)

print(hourly_avg_co2)


fractional_hourly_avg_co2 = hourly_avg_co2

min_value = fractional_hourly_avg_co2.min().min()
max_value = fractional_hourly_avg_co2.max().max()


print(fractional_hourly_avg_co2)
normalized_fractional_hourly_avg_co2 = (fractional_hourly_avg_co2 - min_value) / (max_value - min_value)
normalized_fractional_hourly_avg_co2 = normalized_fractional_hourly_avg_co2.round(2)

print(normalized_fractional_hourly_avg_co2)


############# Ventilation power consumption janurary
"""
Download the data from remoni over jan. This is the apparent power [VA]
Get the sum over the month then multiply with the reaktive power of 0.9
"""

filepath_ven = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Ventilation_loft_jan.csv"

ven_jan = pd.read_csv(filepath_ven)

power_jan = ven_jan[ven_jan['Data type'] == 'Hourly apparent-power'].copy()


power_jan['Timestamp'] = pd.to_datetime(power_jan['Timestamp'])


total_power = power_jan['Value'].sum()*0.9

total_power = total_power.round(2)

print("total power of the ventilation system at ReMoni {} [W]",total_power)