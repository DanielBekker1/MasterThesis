
import pandas as pd
import matplotlib.pyplot as plt


csv_file_path = r"C:\Users\bruger\OneDrive\Dokumenter\MasterThesis\eplusssz.csv"
csv_file_path_2 = r"C:\Users\bruger\OneDrive\Dokumenter\MasterThesis\Test\dataframe_output.csv"

plot_data = pd.read_csv(csv_file_path)  
output_dfs = pd.read_csv(csv_file_path_2)
plt.close

# print("Column Names:")
# print(plot_data.columns.tolist())

# print(plot_data.head())

# plt.figure(figsize=(10, 6))
# plt.plot(plot_data["AIR LOOP HVAC 1:DesPer56:Des Heat Mass Flow [kg/s]"], marker='o', linestyle='-')

# plt.title('AIR LOOP HVAC 1:DesPer56:Des Heat Mass Flow [kg/s]')

# # plt.ylabel('AIR LOOP HVAC 1:DesPer56:Des Heat Mass Flow [kg/s]')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()

su = sum(output_dfs['air_loop_fan_electric_power'])/1000/277.7777777778
print("Sum of the fan electricity{} [W]",su)

df_var = output_dfs

start_point = 0 #30*7 * 24 * 6
num_data_points = start_point + (7*24*6)

week_data = df_var.iloc[start_point:num_data_points]
# print(week_data.shape)
# print(week_data.head())

### Plots for the room temperatures
# fig, (ax1, ax5) = plt.subplots(ncols=2, figsize=(12, 12))  # Remember to change the ncols, number ax and figsize.
# week_data.plot(y='zn_soft1_temp', ax=ax1, color='red')
# #week_data.plot(y='zn_finance1_temp', ax=ax2, color='red')
# week_data.plot(y='zn_hardware_corri_temp', ax=ax5, color='red')
# ax1.set_xlabel('Time')
# ax1.set_title('zn_soft1_temp')


# Plots for the fan mass flow and electricity consumption
# fig, (ax3, ax4) = plt.subplots(ncols=2, figsize=(12, 12))

# week_data.plot(y='air_loop_fan_mass_flow', ax=ax3, color='green')
# week_data.plot(y='air_loop_fan_electric_power', ax=ax4, color='blue')

# ax3.set_ylabel('Fan Mass Flow Rate (kg/s)', color='green')
# ax3.set_xlabel('Time')

# ax4.set_ylabel('Fan Electricity power [W]', color='blue')
# ax4.set_xlabel('Time')
# ax3.legend(loc='upper left')
# ax4.legend(loc='upper left')

# Plot of the out door temperature
# fig, (ax7) = plt.subplots(ncols = 1, figsize=(12,6))
# week_data.plot(y ='oa_db', ax = ax7)
# plt.ylabel('Out Door Temperature [C]')


# Plot the CO2 concentration indoor and outdoor

fig, (ax8) = plt.subplots(ncols = 1, figsize = (12,6))
week_data.plot(y = "Indoor_CO2_zn0", ax = ax8, color = "red")
week_data.plot(y = "air_loop_fan_electric_power", ax = ax8, color = "blue")


fig, (ax9) = plt.subplots(ncols = 1, figsize = (12,6))
week_data.plot(y = "Indoor_CO2_zn2", ax = ax9, color = "blue")
#week_data.plot(y = 'CO2_indoor_con', ax = ax9, color='blue')

ax8.set_xlabel('Time')
ax8.set_ylabel('CO2 concentration [ppm]', color = 'black')

ax9.set_ylabel('CO2 concentration [ppm]', color = 'black')


fig, (ax10) = plt.subplots(ncols = 1, figsize = (12,6))
week_data.plot(y = "Occupancy_schedule", ax = ax10, color = "red")
#week_data.plot(y = 'CO2_indoor_con', ax = ax9, color='blue')

ax10.set_xlabel('Time')
ax10.set_ylabel('infil', color = 'black')

plt.show()


