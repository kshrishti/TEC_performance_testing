import csv
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

FILENAME = 'channel4.csv'
TEMP_HEADER = 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_4.actual.value'
TIME_HEADER = 'Year/Month/Day Hour:Minute:Second.Millisecond'
ABS_THRESHOLD_GRAD = -0.5

def find_step_avg_temp(filename):
	# define the initial step temp and step time
	df = pd.read_csv(filename)

	temps = df[TEMP_HEADER].to_numpy()
	times = df[TIME_HEADER].to_numpy()

	# temps we'll average for each step
	temperatures = []
	step_mean_temps = []


	init_step_idx = 2
	final_step_idx = init_step_idx

	for i in range(2, temps.size):
		temperatures.append(temps[i])
		temp_diff = float(temps[i]) - float(temps[i-1])
		time_1 = datetime.strptime(str(times[i]), "%Y/%m/%d %H:%M:%S")
		time_2 = datetime.strptime(str(times[i-1]), "%Y/%m/%d %H:%M:%S")
		time_diff = (time_1 - time_2).total_seconds()

		grad = temp_diff / time_diff
		next_grad = abs(temps[i+1] - temps[i])

	# if the slope is smaller than the threshold, that means the graph has reached its step
	# find the average of the temperature in that step
		if (grad < ABS_THRESHOLD_GRAD) and (next_grad > ABS_THRESHOLD_GRAD):
			final_step_idx = i
			avg_temp = np.average(temperatures[init_step_idx:final_step_idx])
			avg_time = np.average(times[init_step_idx:final_step_idx])

			step_mean = {}
			step_mean['Time': avg_time]
			step_mean['Average Temperature': avg_temp]
			step_mean_temps.append(step_mean)

			init_step_idx = final_step_idx

	field_names = ['Time', 'Average Temperature']
	with open('Step_Info.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames = field_names)
		writer.writeheader()
		writer.writerows(step_mean_temps)

	return step_mean_temps


def read_power_from_elog(elog_filename):
	power_times = []
	with open(elog_filename) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			power = float(row['Current']) * float(row['Voltage'])
			time = row['Time']
			power_times.append({'Power': power, 'Time': time})

	return power_times


# datetime_lst here is a list of dictionaries with key 'Time'
def convert_datetime_to_onlytime(datetime_lst):
	fixed_lst = []
	for i in range(len(datetime_lst)):
		time = (datetime_lst[i])['Time']
		words = time.split()
		req_time = words[-1]
		other_column = ((datetime_lst[i]).keys())[1]
		fixed_lst.append({'Time': req_time, other_column: (datetime_lst[i])[other_column]})
	return fixed_lst


def compare_temp_power_dependencies(filename, elog_filename):
	# the timestamp here includes only the HH:MM time
	time_powers = read_power_from_elog(elog_filename)
	# the timestamp here includes YY-MM-DD HH:MM:SS
	time_temps = find_step_avg_temp(filename)
	# fix the format to make time_temps only HH:MM:SS
	time_temps = convert_datetime_to_onlytime(time_temps)

	# new list of temperatures and powers
	temps = []
	powers = []
	# get closest time_power to the time_temp and then compare them - if they have the same minute
	for i in range(len(time_powers)):
		time_stamp_power = datetime.strptime((time_powers[i])['Time'], "%H:%M")
		for j in range(len(time_temps - 1)):
			time_stamp_temp_1 = datetime.strptime((time_temps[j])['Time'], "%H:%M:%S")
			time_stamp_temp_2 = datetime.strptime((time_temps[j + 1])['Time'], "%H:%M:%S")
			if (time_stamp_power > time_stamp_temp_1) and (time_stamp_power < time_stamp_temp_2):
				temps.append((time_temps[j])['Average Temperature'])
				powers.append((time_powers[i])['Power'])
	
	temps_powers = [temps, powers]
	return temps_powers


def plot_temp_power_dependencies(temps_powers):
	x_axis = temps_powers[0]
	y_axis = temps_powers[1]

	plt.plot(x_axis, y_axis)
	plt.title('Temperature vs Power')
	plt.xlabel('Average Temperature (ºC)')
	plt.ylabel('Power (W)')
	plt.show()

# if __name__ == '__main__':
# 	find_step_avg_temp(FILENAME)


