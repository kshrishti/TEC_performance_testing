import csv
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import correlate, find_peaks, correlation_lags
from math import sqrt

TIME_HEADER = 'Year/Month/Day Hour:Minute:Second.Millisecond'

CO2_SUPPLY_FILENAME = 'CO2_Supply_TIF_Plant_6_TT7628_4.csv'
CO2_PLANT_FILENAME = 'CO2_Plant_TIF_Plant_6_TT7648_4.csv'
TRAY_PLATE_FILENAME = 'Tray_Plate_TIF_34_2_5_4.csv'
MODULE_FILENAME = 'module_TIF_18_0_2_4.csv'

CO2_SUPPLY_HEADER = 'TIF/Manifold/Loop6/TT7628.actual.floatValue'
MODULE_HEADER = 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value'
CO2_PLANT_HEADER = 'TIF/Manifold/Loop6/TT7648.actual.floatValue'
TRAY_PLATE_HEADER = 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_2/Channel_5.actual.value'

ELOG_FILENAME = '28Jun_Elog.csv'

THRESHOLD = 1.27
MODULE_PLANT_THRESHOLD = 0.7
STEP_THRESHOLD = 0.00009
FALL_THRESHOLD = 10000
WINDOW = "1T" # 1 minute

# FILENAMES = [CO2_SUPPLY_FILENAME, CO2_PLANT_FILENAME, TRAY_PLATE_FILENAME, MODULE_FILENAME]
# HEADERS = [CO2_SUPPLY_HEADER, CO2_PLANT_HEADER, TRAY_PLATE_HEADER, MODULE_HEADER]

FILENAMES = [CO2_PLANT_FILENAME, TRAY_PLATE_FILENAME]
HEADERS = [CO2_PLANT_HEADER, TRAY_PLATE_HEADER]

FILENAME_DICT = {CO2_SUPPLY_FILENAME: CO2_SUPPLY_HEADER, CO2_PLANT_FILENAME: CO2_PLANT_HEADER, TRAY_PLATE_FILENAME: TRAY_PLATE_HEADER, MODULE_FILENAME: MODULE_HEADER}

COLORS = ['blue', 'red', 'green', 'yellow']


def plot_all_temps(filenames, headers):
	for i in range(len(filenames)):
		data = pd.read_csv(filenames[i])
		temps = data[headers[i]].to_numpy()
		temps = temps[1:500]
		temps = temps.astype(float)

		times = data[TIME_HEADER].to_numpy()
		times = times[1:500]
		time = [datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f") for ts in times]

		plt.plot(time, temps, color=COLORS[i], label=filenames[i])

	plt.xlabel('Time')
	plt.ylabel('Temperatures')
	plt.title('All temperatures')
	plt.legend()
	plt.xticks(rotation=45)
	plt.show()

# reformats - resamples, interpolates, formats time as datetime objects
def reformat_df(filename, temp_header):
	df = pd.read_csv(filename)

	temps = df[temp_header].to_numpy()
	temps = temps[1:]
	temps = temps.astype(float)
	temps = pd.Series(temps)

	times = df[TIME_HEADER].to_numpy()
	times = times[1:]
	timestamps = pd.to_datetime(times)
	timestamps = pd.Series(timestamps)

	df = pd.concat([timestamps,temps],axis=1)
	df.columns = [TIME_HEADER, temp_header]
	df[TIME_HEADER] = pd.to_datetime(df[TIME_HEADER], format='%Y/%m/%d %H:%M:%S.%f')
	df.set_index(TIME_HEADER, inplace=True)

	sampling_rate = '10S'  
	resampled_df = df.resample(sampling_rate).mean().interpolate()
	return resampled_df


def find_rise_idx(df, temp_header, threshold):
	temps = df[temp_header]
	rise_index = -1

	for i in range(len(temps)):
		if (temps[i] - temps[0]) > threshold:
			rise_index = i - 2
			break

	if rise_index != -1:
		return rise_index
	else:
		print("No rise detected")
		return -1

	
def find_fall_idx(df, temp_header, threshold):
	temps = df[temp_header]
	fall_index = -1

	for i in range(len(temps)):
		if (temps[i] - temps[0]) < (-threshold):
			fall_index = i - 2
			break

	if fall_index != -1:
		return fall_index
	else:
		print("No fall detected")
		return -1


# find oscillation frequency of plant and plate, assuming the data is already formatted
def find_oscillation_freq(df, temp_header):
	temps = df[temp_header]

	new_times = (df.index).to_numpy()
	new_temps = (df[temp_header]).to_numpy()

	peak_indices, properties = find_peaks(new_temps)

	peaks = new_times[peak_indices]

	oscillation_period = np.mean(np.diff(peaks))
	
	return df, oscillation_period


def start_end_same_timestamp(df1, df2):
    common_start = max(df1.index.min(), df2.index.min())
    common_end = min(df1.index.max(), df2.index.max())

    df1_new = df1.loc[common_start:common_end]
    df2_new = df2.loc[common_start:common_end]

    return df1_new, df2_new


# mean - moving average
def find_moving_average(plant_df, plant_header, window):
	# window = pd.DateOffset(minutes=WINDOW)
	# window = '1T'
	moving_averages = plant_df[plant_header].rolling(window).mean()
	return np.array(moving_averages)


# TODO: check how to perfect
# def find_intersection_indices(plate_df, plate_header, plant_df, plant_header, module_df, module_header):
	# df2 must be decreasing in the intersection period - which is the case when df2 is the plant or the module
	intersection_start_indices = []
	intersection_end_indices = []

	plate_temps = plate_df[plate_header].to_numpy()
	plant_temps = plant_df[plant_header].to_numpy()
	module_temps = module_df[module_header].to_numpy()

	# find where the plant and plate functions intersect
	for i in range(1, len(plate_df) - 1):
		if ((plate_temps[i] >= plant_temps[i]) and (plate_temps[i-1] <= plant_temps[i-1])):
			intersection_start_indices.append(i)
		elif ((plate_temps[i] <= plant_temps[i]) and (plate_temps[i-1] >= plant_temps[i-1])):
			intersection_end_indices.append(i)

	return intersection_start_indices, intersection_end_indices
def find_intersection_indices(supply_df, supply_header, plate_df, plate_header):
	plate_temps = plate_df[plate_header].to_numpy()
	supply_temps = supply_df[supply_header]

	intersection_start_indices = []
	intersection_end_indices = []

	for i in range(1, len(supply_df) - 1):
		if ((plate_temps[i] >= supply_temps[i]) and (plate_temps[i-1] <= supply_temps[i-1])):
			intersection_start_indices.append(i)
		elif ((plate_temps[i] <= supply_temps[i]) and (plate_temps[i-1] >= supply_temps[i-1])):
			intersection_end_indices.append(i)

	return intersection_start_indices, intersection_end_indices

# TODO: when the heaters are turned on, i.e., plant temperature becomes less than the tray temperature, 
# find a different correlation & correction
# assumes the dfs are reformatted already, and start at the same point
def measure_second_correlation(plant_df, plant_header, plate_df, plate_header):
	# if the plant temperature begins to dip below the plate temperature - when heater is turned on - then we have to find second correction
	threshold = 1.5 	# temperature threshold to detect flat top
	window_size = 3
	to_return = []

	# intersection_start_idx = -1
	# intersection_end_idx = -1

	plateau_start_index = -1
	plateau_end_index = -1

	plant_temps = plant_df[plant_header].to_numpy()
	plate_temps = plate_df[plate_header].to_numpy()

	# find where the plant and plate functions intersect
	# for i in range(0, len(plant_df)):
	# 	if (plant_temps[i] <= plate_temps[i]): # and (plant_temps[i-1] >= plate_temps[i-1])):
	# 		intersection_start_idx = i
	# 		break
	# for j in range(intersection_start_idx, len(plate_df)):
	# 	if (plant_temps[j] >= plate_temps[j]): # and (plant_temps[i-1] <= plate_temps[i-1])):
	# 		intersection_end_idx = j
	# 		break
	# print(f"intersection_start: {intersection_start_idx}, intersection_end: {intersection_end_idx}")
	
	intersection_start_idx, intersection_end_idx = find_intersection_indices(plant_df, plant_header, plate_df, plate_header)

	# within this intersection period, find the plateau
	for (df, header) in [(plant_df, plant_header), (plate_df, plate_header)]:
		# selecting the rows between the two indices, and all the columns
		df = df.iloc[intersection_start_idx : intersection_end_idx + 1]
		smoothed_values = df[header].rolling(window_size).mean()
		for i in range(1, len(smoothed_values)):
			diff = abs(smoothed_values[i] - smoothed_values[i-1])
			if diff < threshold:
				plateau_start_index = i
				break
		for j in range(plateau_start_index, len(smoothed_values)):
			diff = abs(smoothed_values[j] - smoothed_values[j-1])
			if diff > threshold:
				plateau_end_index = j
				break
		print(f"plat_start: {plateau_start_index}, plat_end: {plateau_end_index}")
		plateau_df = df.iloc[plateau_start_index:plateau_end_index]

    # Within this intersection period, find the plateau for each DataFrame
		# smoothed_values = df[header].rolling(window_size).mean()
		# plateau_start_index = np.argmax(abs(smoothed_values.diff()) < threshold) + 1
		# plateau_end_index = np.argmax(abs(smoothed_values.diff()) > threshold) + 1
        
		to_return.append((plateau_df, header))

	# We have to_return = [(plant_plateau_df, plant_header), (plate_plateau_df, plate_header)]
	return measure_correlation(to_return[0][0], to_return[0][1], to_return[1][0], to_return[1][1])


# def correcting_final_temp(noisy_filename1, noisy_header1, signal_filename2, signal_header2):
# TODO: need to use the corr_sig to correct the actual signal - subtract the deviations of corr_sig from sig
def correcting_init_temp(noisy_file_df, noisy_header1, signal_file_df, signal_header2, threshold):
	# df1 = reformat_df(noisy_filename1, noisy_header1)
	# df2 = reformat_df(signal_filename2, signal_header2)
	df1 = noisy_file_df
	df2 = signal_file_df
	rise_idx_1 = find_rise_idx(df1, noisy_header1, threshold)
	rise_idx_2 = find_rise_idx(df2, signal_header2, threshold)
	cut_noise = df1.head(rise_idx_1)
	cut_signal = df2.head(rise_idx_2)

	# cut_noise, oscillation_1 = find_oscillation_freq(noisy_filename1, noisy_header1, THRESHOLD)
	# cut_signal, oscillation_2 = find_oscillation_freq(signal_filename2, signal_header2, THRESHOLD)
	# print(f"osc1: {oscillation_1}, osc2: {oscillation_2}")
	
	cut_noise, cut_signal = start_end_same_timestamp(cut_noise, cut_signal)

	sig = cut_signal[signal_header2]
	sig_noise = cut_noise[noisy_header1]

	noise_moving_averages = find_moving_average(cut_noise, noisy_header1, '1T')
	noise_deviations = sig_noise - noise_moving_averages
	signal_moving_averages = find_moving_average(cut_signal, signal_header2, '1T')
	signal_deviations = sig - signal_moving_averages

	amplitude_scale_factor = np.max(noise_deviations)/np.max(signal_deviations)
	average_scale_factor = np.mean(noise_deviations)/np.mean(signal_deviations)
	# amplitude_scale_factor = np.mean(plant_deviations/plate_deviations)
	print(f"amplitude scale factor: {amplitude_scale_factor}")
	print(f"average scale factor: {average_scale_factor}")
	
	translation = np.mean(noise_moving_averages - signal_moving_averages)
	print(f"translation: {translation}")

	corr_sig = sig - (noise_deviations/amplitude_scale_factor)
	corr_sig_avg = sig - (noise_deviations/average_scale_factor)

	# plt.plot(sig, label='signal temperature')
	# plt.plot(sig_noise, label='noise temperature')
	# plt.plot(corr_sig, label='corrected signal temperature')
	# plt.plot(corr_sig_avg, label='corrected (average) signal temperature')
	# plt.xlabel('Time')
	# plt.ylabel('Temperatures')
	# plt.title('Corrections')
	# plt.legend()
	# plt.xticks(rotation=45)
	# plt.show()

	return corr_sig


# assuming the dataframe is already formatted
# returns a list of dicts with 'Time:' and 'Average Temperature:' keys
def find_step_avg_temp(plate_df, plate_header, module_df, module_header, supply_df, supply_header):
	start_indices, end_indices = find_intersection_indices(supply_df, supply_header, plate_df, plate_header)

	start_idx, end_idx = start_indices[-1], end_indices[-1]
	subset_module_df = module_df.iloc[start_idx:end_idx + 1]
	# print(subset_module)
	# x = subset_module_df.index
	# y = subset_module_df[module_header]
	# plt.plot(x, y)
	# plt.xlabel("Time")
	# plt.ylabel("Temperature")
	# plt.title("Zoomed temperature steps of module with heater on")
	# plt.legend()
	# plt.show()

	# fall_idx = find_fall_idx(subset_df2, temp_header2, FALL_THRESHOLD)
	# print(f"fall idx: {fall_idx}")
	# subset_df2 = df2.iloc[fall_idx:end_idx + 1]

	temps = subset_module_df[module_header].values
	times = subset_module_df.index
	temperatures = []
	step_mean_temps = []

	init_step_idx = 0
	final_step_idx = init_step_idx

	for i in range(1, len(temps) - 1):
		temperatures.append(temps[i])
		temp_diff = temps[i] - temps[i-1]
		time_diff = (times[i] - times[i-1]).total_seconds()

		grad = temp_diff / time_diff
		next_grad = (temps[i+1] - temps[i])/(times[i+1] - times[i]).total_seconds()
		
		print(times[i], grad)
		
		if (grad < 0) and (next_grad > 0):
			final_step_idx = i
			avg_temp = np.average(temperatures[init_step_idx:final_step_idx])
			# avg_time = np.average(times[init_step_idx:final_step_idx].astype(float))

			step_mean = {}
			step_mean['Time'] = times[i]
			step_mean['Average Temperature'] = avg_temp
			step_mean_temps.append(step_mean)

			init_step_idx = final_step_idx

	return step_mean_temps

# returns a list of dicts with 'Time:' and 'Power:' keys
def read_power_from_elog(elog_filename):
	power_times = []
	with open(elog_filename, encoding='utf-8-sig') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if ((row['Current'] != '') and (row['Voltage'] != '') and (row['Time'] != '')):
				power = float(row['Current']) * float(row['Voltage'])
				time = datetime.strptime(row['Time'], "%Y/%m/%d %H:%M:%S.%f")
			# time = row['Time']
			power_times.append({'Time': time, 'Power': power})
	
	print(f"Power and time: {power_times}")
	return power_times


# datetime_lst here is a list of dictionaries with key 'Time'
def convert_datetime_to_onlytime(datetime_lst):
	fixed_lst = []
	for i in range(len(datetime_lst)):
		time = (datetime_lst[i])['Time']
		words = time.split()
		req_time = words[-1]
		other_column = list(datetime_lst[i].keys())[1]
		fixed_lst.append({'Time': req_time, other_column: (datetime_lst[i])[other_column]})

	print(f"Datetime conversion: {fixed_lst}")
	return fixed_lst


# TODO: CHECK
def compare_temp_power_dependencies(plate_df, plate_header, plant_df, plant_header, module_df, module_header, supply_df, supply_header, elog_filename):
	time_powers = read_power_from_elog(elog_filename)
	time_temps = find_step_avg_temp(plate_df, plate_header, module_df, module_header, supply_df, supply_header)
	print(time_powers, time_temps)
	# new list of temperatures and powers
	temps = []
	powers = []

	corr_sig = correcting_init_temp(plant_df, plant_header, module_df, module_header, THRESHOLD)
	init_temp = np.mean(corr_sig)
	# print(init_temp)

	# get closest time_power to the time_temp and then compare them - if they have the same minute
	for i in range(len(time_powers)):
		time_stamp_power = (time_powers[i])['Time']
		for j in range(len(time_temps)-1):
			time_stamp_temp_1 = (time_temps[j])['Time']
			time_stamp_temp_2 = (time_temps[j+1])['Time']
			# find the closest time 
			if (time_stamp_power >= time_stamp_temp_1) and (time_stamp_power <= time_stamp_temp_2):
				dt = init_temp - (time_temps[j])['Average Temperature']
				temps.append(dt)
				powers.append((time_powers[i])['Power'])
	
	temps_powers = (temps, powers)

	print(f"Temperature: {temps}\n Power: {powers}")
	return temps_powers


def plot_temp_power_dependencies(temps_powers):
	x_axis, y_axis = temps_powers

	plt.plot(x_axis, y_axis)
	plt.title('Temperature of Module vs Power')
	plt.xlabel('Temperature Difference (ºC)')
	plt.ylabel('Power (W)')
	plt.show()


def compare_rms_with_window_size(noisy_filename1, noisy_header1, signal_filename2, signal_header2):
	cut_noise, oscillation_1 = find_oscillation_freq(noisy_filename1, noisy_header1, THRESHOLD)
	cut_signal, oscillation_2 = find_oscillation_freq(signal_filename2, signal_header2, THRESHOLD)
	
	cut_noise, cut_signal = start_end_same_timestamp(cut_noise, cut_signal)

	sig = cut_signal[signal_header2]
	sig_noise = cut_noise[noisy_header1]
	
	windows = []
	rms_noises = []
	rms_signals = []

	for window in range(100):
		noise_moving_averages = find_moving_average(cut_noise, noisy_header1, f'{window}T')
		noise_deviations = sig_noise - noise_moving_averages
		rms_noise = np.max(noise_deviations)/sqrt(2)
		signal_moving_averages = find_moving_average(cut_signal, signal_header2, f'{window}T')
		signal_deviations = sig - signal_moving_averages
		rms_signal = np.max(signal_deviations)/sqrt(2)
		windows.append(window)
		rms_noises.append(rms_noise)
		rms_signals.append(rms_signal)

	plt.plot(windows, rms_noises, label='rms noise deviation')
	plt.plot(windows, rms_signals, label='rms signal deviation')
	plt.xlabel('window size')
	plt.ylabel('rms value')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	# plot_all_temps(FILENAMES, HEADERS)
	plant_df = reformat_df(CO2_PLANT_FILENAME, CO2_PLANT_HEADER)
	plate_df = reformat_df(TRAY_PLATE_FILENAME, TRAY_PLATE_HEADER)
	module_df = reformat_df(MODULE_FILENAME, MODULE_HEADER)
	supply_df = reformat_df(CO2_SUPPLY_FILENAME, CO2_SUPPLY_HEADER)

	# correcting_init_temp(CO2_PLANT_FILENAME, CO2_PLANT_HEADER, TRAY_PLATE_FILENAME, TRAY_PLATE_HEADER, THRESHOLD)

	# print(find_intersection_indices(plate_df, TRAY_PLATE_HEADER, plant_df, CO2_PLANT_HEADER, module_df, MODULE_HEADER))

	# correcting_init_temp(TRAY_PLATE_FILENAME, TRAY_PLATE_HEADER, MODULE_FILENAME, MODULE_HEADER, MODULE_PLANT_THRESHOLD)
	
	# print(find_intersection_indices(plate_df, TRAY_PLATE_HEADER, plant_df, CO2_PLANT_HEADER, module_df, MODULE_HEADER))
	
	
	# we want to find the module's falling arc, which happens in its intersection with the tray's function
	# step_mean_temps = find_step_avg_temp(plate_df, TRAY_PLATE_HEADER, module_df, MODULE_HEADER, supply_df, CO2_SUPPLY_HEADER)
	# step_mean_temps = pd.DataFrame(step_mean_temps)
	# x = step_mean_temps['Time']
	# y = step_mean_temps['Average Temperature']
	# plt.plot(x, y)
	# plt.xlabel("Time")
	# plt.ylabel("Average Temperature")
	# plt.title("Average temperatures calculated for each step of module")
	# plt.legend()
	# plt.show()

	temps_powers = compare_temp_power_dependencies(plate_df, TRAY_PLATE_HEADER, plant_df, CO2_PLANT_HEADER, module_df, MODULE_HEADER, supply_df, CO2_SUPPLY_HEADER, ELOG_FILENAME)
	plot_temp_power_dependencies(temps_powers)

	
	# compare_rms_with_window_size(CO2_PLANT_FILENAME, CO2_PLANT_HEADER, TRAY_PLATE_FILENAME, TRAY_PLATE_HEADER)
	# compare_rms_with_window_size(TRAY_PLATE_FILENAME, TRAY_PLATE_HEADER, MODULE_FILENAME, MODULE_HEADER)


	# fig, (ax_orig, ax_noise, ax_corr, ax_corrected, ax_corr_lags) = plt.subplots(5, 1, figsize=(4.8, 4.8))
	# ax_orig.plot(sig)
	# ax_orig.set_title('Plate temperature fluctuations')
	# ax_orig.set_xlabel('Timestamp')
	# ax_noise.plot(sig_noise)
	# ax_noise.set_title('Plant temperature fluctuations')
	# ax_noise.set_xlabel('Timestamp')
	# ax_corr.plot(lags, corr)
	# ax_corr.set_title('Cross-correlated signal')
	# ax_corr.set_xlabel('Lag')
	# ax_corrected.plot(corr_sig)
	# ax_corrected.set_title('Plate corrected signal')
	# ax_corrected.set_xlabel('Timestamp')
	# ax_corr_lags.plot(lags2, corr2)
	# ax_corr_lags.set_title('Cross-correlated signal')
	# ax_corr_lags.set_xlabel('Lag')
	# ax_orig.margins(0, 0.1)
	# ax_noise.margins(0, 0.1)
	# ax_corr.margins(0, 0.1)
	# ax_corrected.margins(0, 0.1)
	# ax_corr_lags.margins(0, 0.1)
	# fig.tight_layout()
	# plt.show()

	# corr = correlate(plate_df[TRAY_PLATE_HEADER], plant_df[CO2_PLANT_HEADER])
	# lags = correlation_lags(len(plant_df[CO2_PLANT_HEADER]), len(plate_df[TRAY_PLATE_HEADER]))
	# corr /= np.max(corr)

	
	# print(f"oscillation_plant: {oscillation_plant}, oscillation_plate: {oscillation_plate}")
	# offset, scaling_factor = measure_correlation(cut_plant_df, CO2_PLANT_HEADER, cut_plate_df, TRAY_PLATE_HEADER)
	# print(f"offset: {offset}, scaling factor: {scaling_factor}")
	
	# module_df = reformat_df(MODULE_FILENAME, MODULE_HEADER)
	# corrected_module_df = correcting_init_temp(module_df, plant_df, CO2_PLANT_HEADER, plate_df, TRAY_PLATE_HEADER)
	
	# plant_df, plate_df = start_at_same_point(plant_df, plate_df)
	# print(f"plant df: {plant_df}")
	# print(f"plate df: {plate_df}")
	# print(measure_second_correlation(plant_df, CO2_PLANT_HEADER, plate_df, TRAY_PLATE_HEADER))
	
	# ax = corrected_module_df.plot()
	# # ax = plant_df.plot()
	# # plate_df.plot(ax=ax)
	# plt.xlabel('Time')
	# plt.ylabel('Temperatures')
	# # plt.title('Resampled Plant and Plate temperatures')
	# plt.title('Corrected module temperature')
	# plt.legend()
	# plt.xticks(rotation=45)
	# plt.show()


