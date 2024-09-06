import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##################### Use the ReMoni censor "Billing heatmeter MC602" and "PowerMoniMain" and collect the total hourly power of heating and cooling of the entire building. 

csv_file_path1 = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Data_ReMoni\exportData_Heating_nov.csv"
csv_file_path2 = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Data_ReMoni\exportData_Electricity.csv"

csv_file_path_jan = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Data_ReMoni\Electricity_Jan.csv"
csv_file_path_feb = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Data_ReMoni\Electricity_Feb.csv"
csv_file_path_mar = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Data_ReMoni\Electricity_Mar.csv"
csv_file_path_apr = r"C:\Users\danie\OneDrive\Dokumenter\MasterThesis\Data_ReMoni\Electricity_Apr.csv"

Electricity_csv = pd.read_csv(csv_file_path2)
Heating_csv = pd.read_csv(csv_file_path1)
   

Electricity_jan = pd.read_csv(csv_file_path_jan)
Electricity_feb = pd.read_csv(csv_file_path_feb)
Electricity_mar = pd.read_csv(csv_file_path_mar)
Electricity_apr = pd.read_csv(csv_file_path_apr)

Electricity_consumption = Electricity_csv[Electricity_csv['Data type'] == 'Hourly Total Power'].copy()
Heating_consumption = Heating_csv[Heating_csv['Data type'] == 'Hourly Total Power'].copy()


################## Electricity calculation #####################
Electricity_jan = pd.read_csv(csv_file_path_jan)
Electricity_feb = pd.read_csv(csv_file_path_feb)
Electricity_mar = pd.read_csv(csv_file_path_mar)
Electricity_apr = pd.read_csv(csv_file_path_apr)

Electricity_consumption_jan = Electricity_jan[Electricity_jan['Data type'] == 'Hourly Total Power'].copy()
Electricity_consumption_feb = Electricity_feb[Electricity_feb['Data type'] == 'Hourly Total Power'].copy()
Electricity_consumption_mar = Electricity_mar[Electricity_mar['Data type'] == 'Hourly Total Power'].copy()
Electricity_consumption_apr = Electricity_apr[Electricity_apr['Data type'] == 'Hourly Total Power'].copy()

Electricity_consumption_jan = Electricity_consumption_jan.drop(['UnitTypeInput Id', 'UnitTypeInput Name', 'Unit Input Custom Name','Unit Id', 'Unit Name', 'Timestamp', 'Data type'], axis=1)
Electricity_consumption_feb = Electricity_consumption_feb.drop(['UnitTypeInput Id', 'UnitTypeInput Name', 'Unit Input Custom Name','Unit Id', 'Unit Name', 'Timestamp', 'Data type'], axis=1)
Electricity_consumption_mar = Electricity_consumption_mar.drop(['UnitTypeInput Id', 'UnitTypeInput Name', 'Unit Input Custom Name','Unit Id', 'Unit Name', 'Timestamp', 'Data type'], axis=1)
Electricity_consumption_apr = Electricity_consumption_apr.drop(['UnitTypeInput Id', 'UnitTypeInput Name', 'Unit Input Custom Name','Unit Id', 'Unit Name', 'Timestamp', 'Data type'], axis=1)

elec_sum_jan = Electricity_consumption_jan['Value'].sum() / 1000
elec_sum_feb = Electricity_consumption_feb['Value'].sum() / 1000
elec_sum_mar = Electricity_consumption_mar['Value'].sum() / 1000
elec_sum_apr = Electricity_consumption_apr['Value'].sum() / 1000

print("__________")
print('Sum of the electricity in jan is {:.2f} kW'.format(elec_sum_jan), 'and the consumption in GJ is {:.2f} GJ'.format(elec_sum_jan/277.78))
print("__________")
print('Sum of the electricity in feb is {:.2f} kW'.format(elec_sum_feb), 'and the consumption in GJ is {:.2f} GJ'.format(elec_sum_feb/277.78))
print("__________")
print('Sum of the electricity in mar is {:.2f} kW'.format(elec_sum_mar), 'and the consumption in GJ is {:.2f} GJ'.format(elec_sum_mar/277.78))
print("__________")
print('Sum of the electricity in apr is {:.2f} kW'.format(elec_sum_apr), 'and the consumption in GJ is {:.2f} GJ'.format(elec_sum_apr/277.78))



################## Heating calculation #####################
# print(Heating_consumption.head())


#Reshape the files to have 31 days.
Heating_consumption = Heating_consumption.drop(['UnitTypeInput Id', 'UnitTypeInput Name', 'Unit Input Custom Name','Unit Id', 'Unit Name', 'Timestamp', 'Data type'], axis=1)
Heating_consumption.index = np.arange(1, len(Heating_consumption) + 1)


# Drop of specified rows
Heating_consumption = Heating_consumption.drop([720])
print("the shape of the heating dataframe is {}".format(Heating_consumption.shape))

heat_sum = Heating_consumption['Value'].sum() / 1000
print("__________")
first_timestamp = Heating_consumption["Timestamp short (UTC)"].iloc[0]
last_timestamp = Heating_consumption["Timestamp short (UTC)"].iloc[-1]
print("The timestamp of 30 days start at {} and end at {}".format(first_timestamp, last_timestamp))
print('Sum of the heating is {:.2f}'.format(heat_sum), 'and the consumption in GJ is {:.2f} GJ'.format(heat_sum/277.78))
print('Consumption over a year is {:.2f} kW'.format(heat_sum*12), 'and the consumption in GJ is {:.2f} GJ'.format((heat_sum/277.78)*12))

# plt.figure()
# plt.plot(Electricity_consumption['Timestamp'], Electricity_consumption['Value'])
# plt.xlabel('Time')
# plt.ylabel('Power')   
# plt.title('Electricity Consumption')
# plt.show()



############# Only for electricity

