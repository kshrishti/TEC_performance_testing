import csv
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline, BSpline
from math import sqrt, floor, trunc
from sklearn.metrics import r2_score
import re

TIME_HEADER = 'Year/Month/Day Hour:Minute:Second.Millisecond'

DATE_DICT = {
	'JUN_28' : {
		'CO2_PLANT_FILENAME' : 'CO2_Plant_TIF_Plant_6_TT7648_4.csv',
		'CO2_PLANT_HEADER' : 'TIF/Manifold/Loop6/TT7648.actual.floatValue',
		'TRAY_PLATE_FILENAME' : 'Tray_Plate_TIF_34_2_5_4.csv',
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_2/Channel_5.actual.value',
		'MODULE_FILENAME' : 'module_TIF_18_0_2_4.csv',
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value',
		'PERIOD_BOUNDS' : [15, 25],
		'FLAT_PERIOD' : 150,
		'ELOG_FILENAME' : '28Jun_Elog.csv'
		},
	'JUL_20' : {
		'CO2_PLANT_FILENAME' : '20thJul_CO2.csv', 
		'CO2_PLANT_HEADER' : 'TIF/Manifold/Loop6/TT7648.actual.floatValue', 
		'MODULE_FILENAME' : '20thJul_J20.csv', 
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value', 
		'TRAY_PLATE_FILENAME' : '20thJul_plate.csv', 
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value',
		'PERIOD_BOUNDS' : [20, 35],
		'FLAT_PERIOD' : 360,
		'ELOG_FILENAME' : '20Jul_Elog.csv',
		'CU_HOUSING_FILENAME' : '20Jul_Cu.csv',
		'CU_HOUSING_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value',
		'LYSO_FILENAME': '20Jul_LYSO.csv',
		'LYSO_HEADER': 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_1/Channel_3.actual.value'
		},
	'JUL_25' : {
		'CO2_PLANT_FILENAME' : '25thJul_CO2.csv', 
		'CO2_PLANT_HEADER' : 'TIF/Manifold/Loop6/TT7648.actual.floatValue', 
		'MODULE_FILENAME' : '25thJul_J20.csv', 
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value', 
		'TRAY_PLATE_FILENAME' : '25thJul_plate.csv', 
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value',
		'PERIOD_BOUNDS' : [20, 30],
		'FLAT_PERIOD' : 150,
		'ELOG_FILENAME' : '25Jul_Elog.csv',
		'CU_HOUSING_FILENAME' : '25Jul_Cu.csv',
		'CU_HOUSING_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value',
		'LYSO_FILENAME': '25Jul_LYSO.csv',
		'LYSO_HEADER': 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_1/Channel_3.actual.value'
		},
	'JUL_26' :{
		'MODULE_FILENAME' : 'Jul26_J20.csv', 
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value',
		'TRAY_PLATE_FILENAME' : '26thJul_plate.csv', 
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value',
		'PERIOD_BOUNDS' : [20, 30],
		'ELOG_FILENAME' : '26Jul_Elog_1.csv',
		'CU_HOUSING_FILENAME' : '26Jul_Cu.csv',
		'CU_HOUSING_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value',
		'LYSO_FILENAME': '26Jul_LYSO.csv',
		'LYSO_HEADER': 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_1/Channel_3.actual.value'
		},
	'AUG_02' : {
		'MODULE_FILENAME' : '2ndAug_J20.csv', 
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value', 
		'TRAY_PLATE_FILENAME' : '2ndAug_plate.csv', 
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value',
		'PERIOD_BOUNDS' : [20, 30],
		'FLAT_PERIOD' : 150,
		'ELOG_FILENAME' : '2Aug_Elog.csv',
		'CU_HOUSING_FILENAME' : '2Aug_Cu.csv',
		'CU_HOUSING_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value',
		'LYSO_FILENAME': '2Aug_LYSO.csv',
		'LYSO_HEADER': 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_1/Channel_3.actual.value',
		'AMBIENT_FILENAME': '2Aug_ambient.csv', 
		'AMBIENT_HEADER': 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_2/Channel_0.actual.value' # ambient is 33-2-0
		},
	'AUG_10' : {
		'MODULE_FILENAME' : '10Aug_J20.csv',
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value', 
		'ELOG_FILENAME' : '10Aug_Elog.csv',
		'LYSO_FILENAME': '10Aug_J20.csv',
		'LYSO_HEADER': 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_1/Channel_3.actual.value',
		'CU_HOUSING_FILENAME' : '10Aug_J20.csv',
		'CU_HOUSING_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value',
		'TRAY_PLATE_FILENAME' : '10Aug_J20.csv', 
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value'
	},
	'AUG_18' : {
		'MODULE_FILENAME' : 'Aug_18.csv',
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_1/Channel_0.actual.value', # J19 (18-1-0 refurbished)
		'ELOG_FILENAME' : 'Aug18_Elog_cooling.csv', # for cooling cycle
		# 'ELOG_FILENAME' : 'Aug18_Elog_heating.csv', # for heating cycle
		'LYSO_FILENAME': 'Aug_18.csv',
		'LYSO_HEADER': 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_1/Channel_3.actual.value',
		'CU_HOUSING_FILENAME' : 'Aug18_Cu.csv',
		'CU_HOUSING_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value',
		'TRAY_PLATE_FILENAME' : 'Aug_18.csv',
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value'
	},
	'AUG_24' : {
		'MODULE_FILENAME' : 'Aug_24.csv',
		'MODULE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value',
		'ELOG_FILENAME' : 'Aug_24_Elog.csv',
		'CU_HOUSING_FILENAME' : 'Aug_24.csv',
		'CU_HOUSING_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value',
		'TRAY_PLATE_FILENAME' : 'Aug_24.csv',
		'TRAY_PLATE_HEADER' : 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value'
	}
}


FALL_THRESHOLD = 0.15 	# ºC
RISE_THRESHOLD = 0.18 	# ºC
STEP_THRESHOLD = 0.35 	# ºC
STEP_START_THRESHOLD = 1 		# ºC
STEP_END_THRESHOLD = 1.5		# ºC
MODULE_PLANT_THRESHOLD = 0.7 	# ºC
PLATE_PLANT_THRESHOLD = 1.4 	# ºC
WINDOW = "1T" # 1 minute

ELOG_MODULES = {'28Jun_Elog.csv' : 6, '20Jul_Elog.csv' : 4, '25Jul_Elog.csv' : 4, '26Jul_Elog_1.csv' : 4, '2Aug_Elog.csv' : 4, '10Aug_Elog.csv' : 4, 'Aug18_Elog_heating.csv' : 6, 'Aug18_Elog_cooling.csv' : 6}


# TODO: Further improvement to the code - potentially making all the temperatures 'Data' objects so it's
# easier to store information about them and the functions require fewer arguments - but the current implementation is not
# particularly inefficient either
class Data:
	def __init__(self, filename, header):
		self.filename = filename
		self.header = header
		self.df = reformat_df(filename, header)

	def name(self):
		return self.header

	def temperatures(self):
		return self.df[self.header]

	def length(self):
		return len(self.df)

	def rise_index(self, rise_threshold):
		return find_rise_idx(self.df, self.header, rise_threshold)

	def fall_index(self, fall_threshold):
		return find_fall_idx(self.df, self.header, fall_threshold)

	def popt_for_oscillation_model(self, period_bounds):
		return approximate_signal(self.temperatures().to_numpy(), period_bounds)

	def step_indices(self, fall_threshold, rise_threshold, cooling_bool):
		return find_step_avg_temp2(self.df, self.header, fall_threshold, rise_threshold, cooling_bool)[2]

	def step_mean_temps(self, fall_threshold, rise_threshold, cooling_bool):
		return find_step_avg_temp2(self.df, self.header, fall_threshold, rise_threshold, cooling_bool)[0]


# NOTE: Do a search for 'heating cycle' when you are processing data of an annealing cycle that took place after a cooling cycle
	# There are some blocks of code that need to be uncommented in some relevant functions if it is a heating cycle 
		# Those instructions are all outlined clearly in the docstrings and comments in the functions, but they need to be 
		# uncommented manually because it depends on whether the heating cycle:
			# if after a cooling cycle (like our cases) - relevant blocks need to be uncommented
			# independent heating cycle - blocks don't need to be uncommented


def get_mapping_from_header(header):
	"""
	Returns mapping of a channel according to MTRS convention. 
	
	Arguments: header (str) - header of the csv file that corresponds to a channel mapping on the MTRS
								e.g. 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value'
	Returns: mapping (str) - channel mapping for the header
								e.g. 34-1-0
	"""
	mapping_pattern = 'cms_mtdtf_dcs_1:MTRS/MTD_TIF_(\d+)/Chip_(\d+)/Channel_(\d+).actual.value'
	match = re.search(mapping_pattern, header)
	return f'{match.group(1)}-{match.group(2)}-{match.group(3)}'


def plot_everything(df_list, labels):
	"""
	Plots the temperatures of multiple sensors that are inputted as Dataframes. 
	
	Arguments: df_list (list) - list of all the Dataframes that need to be plotted 
	"""
	for i in range(len(df_list)):
		plt.plot(df_list[i].index, df_list[i].iloc[:,0].to_numpy(), label=labels[i])
	plt.xlabel('Time')
	plt.ylabel('Temperature ºC')
	plt.title('All temperatures')
	plt.legend()
	plt.xticks(rotation=45)
	plt.grid()
	plt.minorticks_on()
	plt.show()


def reformat_df(filename, temp_header):
	"""
	Reformats a csv file by resampling and interpolating the temperatures, and also formatting the timestamps 
	as datetime objects. 
	
	Arguments: 
		filename (str) - csv filename
		temp_header (str) - header of the channel that needs to be reformatted as a DataFrame
	Returns:
		resampled_df (DataFrame) - resampled and interpolated temperatures, with index as timestamps
									and no NaN values
	"""
	df = pd.read_csv(filename)

	temps = df[temp_header].to_numpy()
	temps = temps[1:].astype(float)
	temps = pd.Series(temps)

	times = df[TIME_HEADER].to_numpy()
	timestamps = pd.to_datetime(times[1:])
	timestamps = pd.Series(timestamps)

	df = pd.concat([timestamps,temps],axis=1)
	df.columns = [TIME_HEADER, temp_header]
	df[TIME_HEADER] = pd.to_datetime(df[TIME_HEADER], format='%Y/%m/%d %H:%M:%S.%f')
	df.set_index(TIME_HEADER, inplace=True)

	sampling_rate = '10S'  
	resampled_df = df.resample(sampling_rate).mean().interpolate().dropna()
	return resampled_df


def approximate_signal(function, period_bounds):
	"""
	Uses a convolved function of a sinusoidal function and a linear trend line to get a more accurate
	functional fitting of the temperature fluctuations.
	
	Arguments: 
		function (np array) - the temperatures that are the function to be fitted
		period bounds (tuple) - a predicted range for the period with min, max
	Returns: 
		popt (= amp, period, translation, phi, m, k) - the amplitude, period, translation, phase shift, 
														slope and linear translation of the convolution
	"""
	xs = np.arange(len(function))
	min_max = np.array([tup for tup in zip([0, 100], period_bounds, [-50, 0], [15, 50], [0, 1], [-1000, 100])])
	popt, pcov = curve_fit(convolved_function, np.arange(len(function)), function, maxfev=2000, bounds=min_max)
	amp, period, translation, phi, m, k = popt
	# plt.plot(convolved_function(np.arange(len(function)), amp, period, translation, phi, m, k), label='fitted function')

	# plt.plot(np.arange(len(function)), function, label='original function')
	
	# print(f'approximated function: T = {amp:.2f} sin(2π / {period:.2f} (x - {phi:.2f})) + {translation:.2f} + {m:.2f}x + {k:.2f}')
	# print(f'period = {period:.2f} samples = {10 * period:.2f}s = {(10 * period / 60):.2f} min')

	# june 28
	# plt.plot((0.05 * np.sin((xs - 7) * 2 * np.pi/18)-32.05), label='manual fitting') 	# module
	# plt.plot((0.75 * np.sin((xs - 17) * 2 * np.pi/18)-31.25), label='manual fitting') 	# plant
	# plt.xlabel('Sample No.')
	# plt.ylabel('Temperature (ºC)')
	# plt.legend()
	# plt.minorticks_on()
	# plt.show()

	return popt


def approximate_linear_signal(popt, function):
	"""
	Gives a linearised signal for a function with oscillatory behaviour. 
	
	Arguments:
		popt (amp, period, translation, phi, m, k) - outputted from approximate_signal()
		function (array-like) - the values of the function
	Returns:
		(array) - a new linearised function that is modelled as m*x + k + translation
	"""
	x = np.arange(len(function))
	return popt[4] * x + popt[5] + popt[2]


def sin_curve(times, amp, period, translation, phi):	
	"""
	Creates a a sinusoidal function.
	
	Arguments: x-values (times), amplitude, period, translation and phase shift
	Returns: array of sin function
	"""
	return (amp * np.sin(2 * np.pi/period * (times - phi)) + translation)


def convolved_function(x, amp, period, translation, phi, m, k):	
	"""
	Creates a convolution of a sinusoidal function and linear trend line.
	
	Arguments: x-values, amplitude, period, translation and phase shift
	Returns: array of convolution
	"""
	sinusoidal = amp * np.sin(2 * np.pi/period * (x - phi)) + translation
	general_rise = m * x + k
	return sinusoidal + general_rise


CU_TEMPS_FILENAME = 'Cu_housing_temps.csv'


def cu_housing_comparison(cu_housing_filename):
	"""
	Compares the Cu housing temperatures from multiple sensors to see correlations between hotspots and locations
	of sensors. Plots the temperatures on a graph. 
	
	Arguments: cu_housing_filename (str) - CSV file containing the data from all Cu housing sensors
	"""
	df = pd.read_csv(cu_housing_filename)
	for column_header in list(df.columns)[1:]:
		column_df = reformat_df(cu_housing_filename, column_header)
		temps = column_df[column_header].to_numpy()
		times = column_df.index
		try:
			linear_temps = approximate_linear_signal(approximate_signal(temps, [20, 30]), temps)
			plt.plot(times, linear_temps, label=f'linearised {get_mapping_from_header(column_header)}')
		except:
			print('Missing data')
	plt.title('Copper housing sensor temperatures')
	plt.ylabel('Temperature (ºC)')
	plt.xlabel('Time')
	plt.legend()
	plt.minorticks_on()
	plt.show()


def find_rise_idx(df, temp_header, threshold):	
	"""
	Finds the index at which the function in the DataFrame begins to rise above a set threshold; 
	used to find when the SiPMs are turned on, or to find the start of the TEC annealing cycle. 
	
	Arguments: 
		df (DataFrame) - contains function being scanned
		temp_header (str) - header of the df that contains the function to be scanned (channel name)
		threshold (int) - min temp difference that would be considered a "rise" in temperature
	Returns:
		rise_idx (int) - index of function at which the rise occurs
		-1 if no rise found
	"""
	temps = df[temp_header].values
	rise_index = -1

	for i in range(len(temps) - 2):
		if ((temps[i+1] - temps[i]) > threshold) and ((temps[i+2] - temps[i+1]) > 0):
			rise_index = i
			break

	if rise_index != -1:
		return rise_index
	else:
		print("No rise detected")
		return -1


def find_fall_idx(df, temp_header, threshold):
	"""
	Finds the index at which the function in the DataFrame begins to fall below a set threshold; 
	used to find when the cooling cycle of the TECs starts. 
	
	Arguments: 
		df (DataFrame) - contains function being scanned
		temp_header (str) - header of the df that contains the function to be scanned (channel name)
		threshold (int) - min temp difference that would be considered a "rise" in temperature
	Returns:
		rise_idx (int) - index of function at which the rise occurs
		-1 if no rise found
	"""
	temps = df[temp_header]
	fall_index = -1

	for i in range(len(temps) - 2):
		if ((temps[i+1] - temps[i]) < (-threshold)) and ((temps[i+2] - temps[i+1]) < (-threshold)):
			fall_index = i - 2
			break

	if fall_index != -1:
		return fall_index
	else:
		print("No fall detected")
		return -1


def find_oscillation_freq(df, temp_header):	
	"""
	Finds oscillation frequency of plant and plate temperatures, especially for erratic functions (like the CO2
	plant). Instead of using sinusoidal approximation, uses the difference in timings between peaks. 
	
	Arguments: DataFrame and relevant header for the temperature function. 
	Returns: 
		oscillation_period (int) - approximate period of oscillation
		peak_indices (list) - list of indices at which the function peaks
		trough_indices (list) - list of indices at which the function has troughs
	"""
	temps = df[temp_header]

	new_times = (df.index).to_numpy()
	new_temps = (df[temp_header]).to_numpy()

	peak_indices, properties = find_peaks(new_temps)
	trough_indices, properties = find_peaks(-1 * new_temps)

	if (len(peak_indices) == 0) or (len(trough_indices) == 0):
		print("No peaks")
		return [], [], []

	differences = np.diff(peak_indices)
	peak_indices = peak_indices[np.append(True, differences > 2)]

	diffs = np.diff(trough_indices)
	trough_indices = trough_indices[np.append(True, diffs > 7)]

	peaks = new_times[peak_indices]

	oscillation_period = np.mean(np.diff(peaks))
	oscillation_period = oscillation_period.astype(int)

	# plt.plot(new_temps, label='temperatures')
	# plt.plot(peak_indices, new_temps[peak_indices], 'ro')
	# plt.plot(trough_indices, new_temps[trough_indices], 'bo')
	# plt.minorticks_on()
	# plt.show()

	return oscillation_period, peak_indices, trough_indices


def start_end_same_timestamp(df1, df2):
	"""
	Ensures 2 DataFrames start and end at the same timestamp so they can be compared. 
	Redundant function because same_time_frame() exists, but could be more efficient if wanted to reformat only
	2 DataFrames. 
	
	Arguments: 2 DataFrames
	Returns: 2 reformatted DataFrames
	"""
    common_start = max(df1.index.min(), df2.index.min())
    common_end = min(df1.index.max(), df2.index.max())

    df1_new = df1.loc[common_start:common_end]
    df2_new = df2.loc[common_start:common_end]

    return df1_new, df2_new


def same_time_frame(df_list):	
	"""
	Ensures a list of DataFrames start and end at the same timestamp so they can be compared. 
	
	Arguments: a list of DataFrames
	Returns: a list of DataFrames
	"""
	common_start = max(df.index.min() for df in df_list)
	common_end = min(df.index.max() for df in df_list)
	df_list_new = [df.loc[common_start:common_end] for df in df_list]
	return df_list_new


def find_moving_average(df, header, window):
	"""
	Finds the moving average of a function (a header in a DataFrame) with a given window size.
	
	Arguments:
		df (DataFrame)
		header (str)
		window (str) - follows rolling(window) conventions for pandas e.g. '1T' or '30s'
	Returns:
		moving_averages (array)
	"""
	moving_averages = df[header].rolling(window).mean()
	return np.array(moving_averages)


def find_intersection_indices(supply_df, supply_header, plate_df, plate_header):	
	"""
	Find indices of points of intersections of the supply function and the plate function (another [but less
	reliable] way to identify when the TECs are turned on).
	"""
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


def correcting_temp(cut_noise, noisy_header, cut_signal, signal_header):
	"""
	DEFUNCT - use correcting_out_oscillations() instead
	First attempt at correcting the temperature. 
	
	Arguments: cut_noise and cut_signal mean the DataFrames only include the points that need to be corrected out
				(flat parts of the function).
	Returns: 
		deviations from corrected signal, deviations from original signal, approx. translation of sinusoidal function
	"""
	sig = cut_signal[signal_header]
	sig_noise = cut_noise[noisy_header]

	
	window = '161s' # found using scan of rms versus window size
	noise_moving_averages = find_moving_average(cut_noise, noisy_header, window)
	signal_moving_averages = find_moving_average(cut_signal, signal_header, window)

	# translation of functions from y = 0 taken to be the mean of their moving averages
	noise_translation = np.mean(noise_moving_averages)
	noise_deviations = sig_noise - noise_translation
	signal_translation = np.mean(signal_moving_averages)
	signal_deviations = sig - signal_translation

	# calculating rms of the signal with respect to its rolling mean 
	rms_value = -sqrt(np.mean(sig ** 2))
	rolling_mean_squared = (pd.Series(signal_deviations ** 2)).rolling(window).mean()
	new_devs = np.mean(np.sqrt(rolling_mean_squared))

	# scaling down the deviations from the noisy signal, then subtracting those scaled deviations from the main signal
	lag, amplitude_scale_factor = measure_cross_correlation(cut_noise, noisy_header, cut_signal, signal_header)

	corr_rms_value = -sqrt(np.mean(sig ** 2))
	corr_sig = sig - (noise_deviations/amplitude_scale_factor)
	corr_sig_moving_averages = pd.Series(corr_sig).rolling(window).mean()
	corr_sig_deviations = corr_sig - corr_sig_moving_averages
	corr_rolling_mean_squared = (pd.Series(corr_sig_deviations ** 2)).rolling(window).mean()
	corr_new_devs = np.mean(np.sqrt(corr_rolling_mean_squared))

	corr_new_devs = np.sqrt(corr_sig_deviations ** 2)
	new_devs = np.sqrt(signal_deviations ** 2)
	# return corr_sig_deviations, signal_deviations, signal_translation
	return corr_new_devs, new_devs, signal_translation


def correcting_out_oscillations(sig_fn, noise_fn, date_str):	
	"""
	Updated way to correct out oscillations. 
	
	Arguments: signal and noise functions, string date of test (used in DATE_DICT to get information).
	Returns: 
		deviations from corrected signal, deviations from original signal, corrected signal
	"""
	sig_popt = approximate_signal(sig_fn, DATE_DICT[date_str]['PERIOD_BOUNDS'])
	noise_popt = approximate_signal(noise_fn, DATE_DICT[date_str]['PERIOD_BOUNDS'])

	s_amp, s_period, s_translation, s_phi, s_m, s_k = sig_popt

	n_amp, n_period, n_translation, n_phi, n_m, n_k = noise_popt

	# since the phase shift is returned in seconds, but we want it in terms of the samples 
	phase_shift = trunc(measure_phase_shift(noise_popt, sig_popt)/10)
	if phase_shift != 0:
		noise_fn, sig_fn = noise_fn[:-phase_shift], sig_fn[phase_shift:]

	xs = np.arange(len(noise_fn))
	linear_sig = s_m * xs + s_k + s_translation
	linear_noise = n_m * xs + n_k + n_translation

	# getting deviations from the linear trendline
	sig_devs = sig_fn - linear_sig

	noise_deviations = noise_fn - linear_noise

	amplitude_scale_factor = n_amp/s_amp
	# amplitude_scale_factor = noise_deviations/sig_devs

	sinusoidal_sig = s_amp * np.sin(2*np.pi / s_period * (xs - s_phi)) + s_translation + s_m * xs + s_k
	sinusoidal_noise = n_amp * np.sin(2*np.pi / n_period * (xs - n_phi)) + n_translation + n_m * xs + n_k
	# scaled_sinusoidal_noise = (n_amp/amplitude_scale_factor) * np.sin(2*np.pi / n_period * (xs - n_phi)) + n_translation + n_m * xs + n_k

	corr_sig = sig_fn - (noise_deviations/amplitude_scale_factor)

	# print(f'noise devs/signal devs = {np.mean(noise_deviations/sig_devs)}')
	# print(f'noise amp/signal amp = {n_amp/s_amp}')

	corr_devs = sig_fn - corr_sig


	# corr_rms_value = -sqrt(np.mean(sig_fn ** 2))
	# corr_sig = sig_fn - (noise_deviations/amplitude_scale_factor)
	# corr_sig_moving_averages = pd.Series(corr_sig).rolling(window).mean()
	# corr_sig_deviations = corr_sig - corr_sig_moving_averages
	# corr_rolling_mean_squared = (pd.Series(corr_sig_deviations ** 2)).rolling(window).mean()
	# corr_devs = np.mean(np.sqrt(corr_rolling_mean_squared))

	# plt.plot(noise_fn, label='Plate temperature')
	# plt.plot(linear_noise, label='linear plate temperature')
	# plt.plot(linear_sig, label='linear module temperature')
	# plt.plot(sig_fn, label='Module temperature')
	# plt.plot(corr_sig, label='Corrected module temperature')
	# plt.plot(sig_devs, label='Module temperature deviations')
	# plt.plot(noise_deviations, label='noise deviations')
	# plt.plot((noise_deviations / 7), label='noise devs scaled')
	# plt.plot(sinusoidal_sig, label='Sin module temperature')
	# plt.plot(sinusoidal_noise, label='Sin plate temperature')
	# plt.plot(scaled_sinusoidal_noise, label='scaled sin noise')
	# plt.legend()
	# plt.minorticks_on()
	# plt.show()

	new_corr_devs = np.sqrt(corr_devs ** 2)
	new_devs = np.sqrt(sig_devs ** 2)
	return new_corr_devs, new_devs, corr_sig
	

def correcting_init_temp(noisy_file_df, noisy_header1, signal_file_df, signal_header2, signal_rise_threshold, signal_fall_threshold, date_str):	
	"""
	Locates the segment before SiPM or TEC power is injected, and approximates the initial temperature from that. 
	Uses the corr_sig from correcting_out_oscillations() to correct the actual signal - subtracts the deviations 
	of corr_sig from the actual signal function. 
	
	Arguments:
		noisy_file_df (DF) - DataFrame for noise function (usually taken to be the plate)
		noisy_header1 - header for plate
		signal_file_df - DataFrame for signal (usually taken to be the module)
		signal_header2 - module header
		signal_rise_threshold - threshold for signal rise to be detected
		signal_fall_threshold - threshold for signal fall to be detected
		date_str - string of date (used in DATE_DICT to get more information)
	
	Returns:
		mean initial temperature (float)
		error (int) - error associated with the initial temperature (max corrected dev - min corrected dev)/2
		min_idx (int) - the index of the rise or fall, whose result depends on whether SiPMs were turned on 
						before or after the TECs. 
	"""
	df1 = noisy_file_df
	df2 = signal_file_df
	rise_idx = find_rise_idx(df2, signal_header2, signal_rise_threshold)
	fall_idx = find_fall_idx(df2, signal_header2, signal_fall_threshold)

	# fall first or rise first depends on SiPM load (0 or non-zero respectively)
	min_idx = min(fall_idx, rise_idx)
	cut_signal = df2.head(min_idx)
	cut_noise = df1.head(min_idx)

	if not min_idx:
		print("No rise or fall found")

	# earlier iteration with defunct function, doesn't work as well the updated version
		# corr_devs, init_devs, corr_sig = correcting_temp(cut_noise, noisy_header1, cut_signal, signal_header2)
		# print(f"percent improvement w rolling mean:\n {np.mean((corr_devs - init_devs)/init_devs)}")

	sig_fn = cut_signal[signal_header2].to_numpy()
	noise_fn = cut_noise[noisy_header1].to_numpy()

	alt_corr_devs, alt_init_devs, alt_corr_sig = correcting_out_oscillations(sig_fn, noise_fn, date_str)
	amp, period, translation, phi, m, k = approximate_signal(sig_fn, DATE_DICT[date_str]['PERIOD_BOUNDS'])
	approximate_translation = m * np.arange(len(sig_fn)) + k + translation 	# same thing as approximate_linear_signal()
	
	# print(f"percent improvement w functional fitting:\n {np.mean((alt_corr_devs - alt_init_devs)/alt_init_devs)}")

	error = (np.max(alt_corr_devs) - np.min(alt_corr_devs))/2

	# plt.plot(sig_fn, label='module temperature', color='blue')
	# plt.plot(noise_fn, label='plate temperature', color='orange')
	# plt.axhline(y=corr_sig, label='signal translation rolling', color='green')
	# plt.plot(approximate_translation, label='signal translation convolved', color='black')
	# plt.plot(alt_corr_sig, label='functionally corrected signal temp', color='purple')
	# # plt.plot(signal_moving_averages, label='mean signal temperature (window=161s)')
	# # plt.plot(corrected, label='corrected', color='red')
	# # plt.plot(rolling_rms_value, label='rolling rms of signal', color='pink')
	# # plt.axhline(y=rms_value, label='rms value', color='purple')
	# plt.plot(corr_sig, label='corrected signal temperature', color='green')
	# # plt.plot(corr_rolling_rms_value, label='rolling rms of corr signal', color='pink')
	# # plt.axhline(y=corr_rms_value, label='corr rms value', color='purple')
	# plt.plot(signal_file_df[signal_header2].to_numpy(), label='actual temperatures', color='green')
	# plt.axhline(y=np.mean(alt_corr_sig), label='init temperature', color='black', linestyle='--')
	# plt.axhline(y=noise_translation, label='noise translation', color='gray', linestyle='--')

	# # plt.plot(signal_deviations, label='module deviations from rolling mean')
	# # plt.plot(noise_deviations, label='plate deviations from rolling mean')
	# plt.xlabel('Samples')
	# plt.ylabel('Temperatures')
	# plt.title('Corrections')
	# plt.legend()
	# plt.xticks(rotation=45)
	# plt.minorticks_on()
	# plt.show()

	return np.mean(alt_corr_sig), error, min_idx


def scanning_for_window_size(noisy_file_df, noisy_header1, signal_file_df, signal_header2, noise_rise_threshold, signal_rise_threshold):	
	"""
	DEFUNCT
	To find the best windox size to use in a rolling mean of the temperature functions. 
	"""
	df1 = noisy_file_df
	df2 = signal_file_df
	rise_idx_1 = find_rise_idx(df1, noisy_header1, noise_rise_threshold)
	rise_idx_2 = find_rise_idx(df2, signal_header2, signal_rise_threshold)
	# cuts off at the first rise
	cut_noise = df1.head(rise_idx_1)
	cut_signal = df2.head(rise_idx_2)
	cut_noise, cut_signal = start_end_same_timestamp(cut_noise, cut_signal)

	sig = cut_signal[signal_header2]
	sig_noise = cut_noise[noisy_header1]

	windows = np.arange(2000)

	best_window = 0
	improvement = 0

	for window_size in windows:
		window = f'{window_size}s'
		# finding the frequencies of both signals so they can be used for the window sizes for the rolling mean
		noise_osc_freq, noise_peak_indices, noise_troughs = find_oscillation_freq(cut_noise, noisy_header1)
		noise_moving_averages = find_moving_average(cut_noise, noisy_header1, window)
		noise_deviations = sig_noise - noise_moving_averages

		signal_osc_freq, signal_peak_indices, signal_troughs = find_oscillation_freq(cut_signal, signal_header2)
		signal_moving_averages = find_moving_average(cut_signal, signal_header2, window)
		signal_deviations = sig - signal_moving_averages

		# translation of functions from y = 0 taken to be the mean of their moving averages
		noise_translation = np.mean(noise_moving_averages)
		signal_translation = np.mean(signal_moving_averages)

		# # calculating rms with respect to the rolling mean
		# rms_value = -sqrt(np.mean(sig ** 2))
		# rolling_mean_squared = (pd.Series(signal_deviations ** 2)).rolling(window).mean()
		# rolling_rms_value = np.sqrt(rolling_mean_squared) + signal_translation
		# new_devs = np.mean(np.sqrt(rolling_mean_squared))

		signal_deviations = sig - signal_translation
		noise_deviations = sig_noise - noise_translation

		# calculating rms with respect to the translation
		rms_value = -sqrt(np.mean(sig ** 2))
		rolling_mean_squared = (pd.Series(signal_deviations ** 2)).rolling(window).mean()
		rolling_rms_value = np.sqrt(rolling_mean_squared) + signal_translation
		new_devs = np.mean(np.sqrt(rolling_mean_squared))

		# scaling down the deviations from the noisy signal, then subtracting those scaled deviations from the main signal
		lag, amplitude_scale_factor = measure_cross_correlation(cut_noise, noisy_header1, cut_signal, signal_header2)

		corr_sig = sig - (noise_deviations/amplitude_scale_factor)
		corr_sig_moving_averages = pd.Series(corr_sig).rolling(window).mean()
		corr_sig_deviations = corr_sig - corr_sig_moving_averages
		corr_signal_translation = np.mean(corr_sig_moving_averages)

		corr_rms_value = -sqrt(np.mean(sig ** 2))
		corr_rolling_mean_squared = (pd.Series(corr_sig_deviations ** 2)).rolling(window).mean()
		corr_rolling_rms_value = np.sqrt(corr_rolling_mean_squared) + corr_signal_translation
		corr_new_devs = np.mean(np.sqrt(corr_rolling_mean_squared))

		new_improvement = (corr_new_devs - new_devs)/new_devs
		if abs(new_improvement) > abs(improvement):
			improvement = new_improvement
			best_window = window

	plt.plot(sig, label='signal temperature', color='blue')
	plt.plot(sig_noise, label='noise temperature', color='orange')
	# plt.plot(signal_moving_averages, label='mean signal temperature (window=161s)')
	# plt.plot(rolling_rms_value, label='rolling rms of signal', color='pink')
	# plt.axhline(y=rms_value, label='rms value', color='purple')
	plt.plot(corr_sig, label='corrected signal temperature', color='green')
	# plt.plot(corr_rolling_rms_value, label='rolling rms of corr signal', color='pink')
	# plt.axhline(y=corr_rms_value, label='corr rms value', color='purple')

	# plt.plot(signal_deviations, label='module deviations from translation')
	# plt.plot(noise_deviations, label='plate deviations from translation')
	plt.xlabel('Time')
	plt.ylabel('Temperatures')
	plt.title('Corrections')
	plt.legend()
	plt.xticks(rotation=45)
	plt.minorticks_on()
	plt.show()

	return best_window, improvement
	

def read_power_from_elog(elog_filename):	
	"""
	Calculates the power per SiPM array (4 TECs) for an inputted elog filename. Number of modules are gotten
	from the ELOG_MODULES dictionary. 
	
	Arguments: elog_filename (str) - filename of the CSV file with the elog with headers 'Time', 'Current' 
										and 'Voltage'
	
	Returns:
		resampled_power_time (DF) - timestamps as index column, TEC powers are resampled to reflect 
									continuous power supply
		power_time (DF) - timestamps as index column, TEC powers are other column
	"""
	num_modules = ELOG_MODULES[elog_filename]
	power_times = []
	with open(elog_filename, encoding='utf-8-sig') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if ((row['Current'] != '') and (row['Voltage'] != '') and (row['Time'] != '')):
				# power in mW
				power = float(row['Current']) * float(row['Voltage']) / (4 * num_modules * 0.001)
				try:
					time = datetime.strptime(row['Time'], "%H:%M:%S")
				except:
					time = datetime.strptime(row['Time'], "%H:%M")
			# time = row['Time']
				power_times.append({'Time': time, 'Power': power})
	
	power_time = pd.DataFrame.from_dict(power_times)
	power_time['Time'] = pd.to_datetime(power_time['Time'])
	power_time.set_index('Time', inplace=True)


	# print(f"Power time: {power_time}")
	sampling_rate = '10S'  
	resampled_power_time = power_time.resample(sampling_rate).mean().interpolate(method='pad')
	resampled_power_time = resampled_power_time.interpolate(limit_area='outside')
	# print(f"Resampled Power and time: {resampled_power_time}")
	# plt.title("Power vs time")
	# plt.xlabel("Time")
	# plt.ylabel('Power (mW)')
	# plt.plot(resampled_power_time, label='Interpolated (with steps) and resampled power')
	# plt.minorticks_on()
	# plt.show()
	return resampled_power_time, power_time


def read_current_from_elog(elog_filename):
	"""
	Returns total current (as a list) from the elog file (CSV) at each TEC power step. 
	"""
	currents = []
	with open(elog_filename, encoding='utf-8-sig') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if ((row['Current'] != '') and (row['Voltage'] != '') and (row['Time'] != '')):
				# current in A
				currents.append(float(row['Current']))
	return currents


def read_voltage_from_elog(elog_filename):	
	"""
	Returns total voltage (as a list) from the elog file (CSV) at each TEC power step. 
	"""
	voltages = []
	with open(elog_filename, encoding='utf-8-sig') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if ((row['Current'] != '') and (row['Voltage'] != '') and (row['Time'] != '')):
				# current in A
				voltages.append(float(row['Voltage']))
	return voltages


def find_step_avg_temp(plate_df, plate_header, module_df, module_header, supply_df, supply_header, elog_filename):
	"""
	DEFUNCT - Use find_step_avg_temp2() instead. 
	Finds the average temperatures for each step of TEC power. 
	
	Arguments:
		plate_df and plate_header - for cold plate temperatures
		module_df and module_header - for module temperatures
		supply_df and supply_header - for CO2 inlet temperature 
	
	Returns:
		1. a list of dicts with 'Time' and 'Average Temperature' keys
		2. an array of the uncertainties in the average
		3. a tuple of the (step start indices, step end indices)
	"""
	start_indices, end_indices = find_intersection_indices(supply_df, supply_header, plate_df, plate_header)

	start_idx, end_idx = start_indices[-1], end_indices[-1]
	print(f"start idx: {start_idx}")
	subset_module_df = module_df.iloc[start_idx:end_idx + 1]

	temps = subset_module_df[module_header].to_numpy()
	times = subset_module_df.index
	temperatures = []
	step_mean_temps = []

	plot_times = []
	plot_temps = []

	init_step_idx = 0
	final_step_idx = 0

	step_start_indices = []
	step_end_indices = []
	errors = [0.1717936197916643]
	interpolated_power_times, power_times = read_power_from_elog(elog_filename)
	falling_timestamps = power_times.index


	for i in range(1, len(temps) - 1):
		temperatures.append(temps[i])
		temp_diff = temps[i] - temps[i-1]
		time_diff = (times[i] - times[i-1]).total_seconds()

		grad = temp_diff / time_diff

		next_temp_diff = (temps[i+1] - temps[i])
		next_time_diff = (times[i+1] - times[i]).total_seconds()
		next_grad = next_temp_diff / next_time_diff
		

		diff_from_prev_step = temps[final_step_idx] - temps[i]

		# if (final_step_idx == 0):
		# 	if ((grad >= 0) and (next_grad < 0) and (next_temp_diff < -1)):
		# 		final_step_idx = i
		# 		final_step_idx = i
		# 		kink_start_indices.append(i)
		# 		avg_temp = np.average(temperatures[init_step_idx:final_step_idx])

		# 		# reference_time = np.datetime64(times[init_step_idx])
		# 		# avg_time = reference_time + np.sum([(date - reference_time) for date in times[init_step_idx:final_step_idx]], timedelta()) / (final_step_idx + 1 - init_step_idx)
		# 		avg_time = times[floor((init_step_idx+final_step_idx)/2)]

		# 		step_mean = {}
		# 		step_mean['Time'] = avg_time
		# 		step_mean['Average Temperature'] = avg_temp
		# 		step_mean_temps.append(step_mean)

		# 		init_step_idx = final_step_idx
		# else:
		# if ((grad < 0) and (next_grad >= 0) and (next_temp_diff < 1)):
		if ((grad > next_grad) and (temps[init_step_idx] - temps[i] > 1)):
			init_step_idx = i
			step_start_indices.append(i)
		elif ((grad >= 0) and (next_grad < 0) and (diff_from_prev_step > 1.5)):
			final_step_idx = i
			step_end_indices.append(i)
			avg_temp = np.average(temperatures[init_step_idx:final_step_idx])
			# reference_time = np.datetime64(times[init_step_idx])
			# avg_time = reference_time + np.sum([(date - reference_time) for date in times[init_step_idx:final_step_idx]], timedelta()) / (final_step_idx + 1 - init_step_idx)
			avg_time = times[floor((init_step_idx+final_step_idx)/2)]
			step_mean = {}
			step_mean['Time'] = avg_time
			plot_times.append(avg_time)
			step_mean['Average Temperature'] = avg_temp
			plot_temps.append(avg_temp)
			step_mean_temps.append(step_mean)

			init_step_idx = final_step_idx


	for i in range(len(step_start_indices)):
		errors.append((temps[step_start_indices[i]]-temps[step_end_indices[i]])/2)

	x = subset_module_df.index
	y = subset_module_df[module_header]
	plt.plot(x, y)
	plt.scatter(plot_times, plot_temps, color='red', label='average time and step temp')
	plt.scatter(times[step_end_indices], temps[step_end_indices], color='yellow', label='step ends')
	plt.scatter(times[step_start_indices], temps[step_start_indices], color='green', label='step starts')
	plt.xlabel("Time")
	plt.ylabel("Temperature")
	plt.title("Zoomed temperature steps of module with falling temperature")
	plt.legend()
	plt.minorticks_on()
	plt.show()
	return step_mean_temps, np.array(errors), (start_idx + np.array(step_start_indices), start_idx + np.array(step_end_indices))


def find_step_avg_temp2(module_df, module_header, fall_threshold, rise_threshold, cooling_bool):	
	"""
	Finds the average temperatures for each step of TEC power. 
	There are two implementations that can be toggled with the cooling_bool - 
		for the TEC cooling cycle (where fall comes before the rise), or
		the TEC annealing cycle (where the rise comes before the fall) 
			- however, it should be noted that if SiPMs are turned on before the TECs in the annealing cycle, 
			that could be detected as the first rise - the code should then be corrected to take the second rise in temperature
	The average temperature is calculated by taking the mean of all the temperatures in the step. 
	
	Arguments:
		module_df and module_header - for module temperatures
		fall_threshold/rise_threshold (int) - what temperature difference counts as a TEC step fall/rise
		cooling_bool (bool) - if True, cooling cycle; if False, heating cycle
	Returns:
		1. step_mean_temps (list) - a list of dicts with 'Time' and 'Average Temperature' keys
		2. an array of the uncertainties in the average
		3. a tuple of the (step start indices, step end indices)
	"""
	# for TEC cooling cycle - fall comes before rise
	if cooling_bool:
		fall_idx = find_fall_idx(module_df, module_header, fall_threshold)
		new_mod_df = module_df.iloc[fall_idx:]
		rise_idx = find_rise_idx(new_mod_df, module_header, rise_threshold)
		new_mod_df = new_mod_df.iloc[:rise_idx]

	# for TEC heating cycle, rise before the fall
	else:
		rise_idx = find_rise_idx(module_df, module_header, rise_threshold)
		new_mod_df = module_df.iloc[rise_idx:]
		fall_idx = find_fall_idx(new_mod_df, module_header, fall_threshold)
		new_mod_df = new_mod_df.iloc[:fall_idx]

	temps = new_mod_df[module_header].to_numpy()
	times = new_mod_df.index
	# This is the temperature right as it falls because of the TECs 
	step_mean_temps = [{'Time':times[0], 'Average Temperature':temps[0]}]

	plot_times = [times[0]]
	plot_temps = [temps[0]]

	step_start_indices = []
	actual_step_start_indices = []
	step_end_indices = []
	actual_step_end_indices = []
	step_indices = []

	errors = []

	for i in range(1, len(temps) - 1):
		temp_diff = temps[i] - temps[i-1]
		time_diff = (times[i] - times[i-1]).total_seconds()

		grad = temp_diff / time_diff

		next_temp_diff = (temps[i+1] - temps[i])
		next_time_diff = (times[i+1] - times[i]).total_seconds()
		next_grad = next_temp_diff / next_time_diff
		

		# this gets all the temperatures in the plateau regions
		if ((grad < next_grad) and (temp_diff < next_temp_diff)):
			step_indices.append(i)


	# removes most of the middle points so you only get the step starts and ends 
	for j in range(1, len(step_indices)):
		idx = step_indices[j]
		prev_idx = step_indices[j-1]
		if (abs(temps[idx] - temps[prev_idx]) > STEP_THRESHOLD): 
			step_start_indices.append(idx)
			step_end_indices.append(prev_idx)

	step_end_indices.append(len(temps) - 1)

	# sometimes there are too many step starts and step ends, so need to recursively go through them
	# to get the ones we want - the actual step starts and ends
	for k in range(len(step_end_indices) - 1):
		for l in range(len(step_start_indices) - 1):
			curr_ss_ts = times[step_start_indices[l]]
			next_ss_ts = times[step_start_indices[l+1]]
			prev_se_ts = times[step_end_indices[k]]
			next_se_ts = times[step_end_indices[k+1]]
			if (curr_ss_ts > prev_se_ts) and (curr_ss_ts < next_se_ts) and (next_ss_ts > next_se_ts):
				actual_step_start_indices.append(step_start_indices[l])
			
	actual_step_start_indices.append(step_start_indices[-1])

	for m in range(len(actual_step_start_indices) - 1):
		for n in range(len(step_end_indices) - 1):
			prev_se_ts = times[step_end_indices[n]]
			curr_se_ts = times[step_end_indices[n+1]]
			prev_ss_ts = times[actual_step_start_indices[m]]
			next_ss_ts = times[actual_step_start_indices[m+1]]
			if (curr_se_ts > prev_ss_ts) and (curr_se_ts < next_ss_ts) and (prev_se_ts < prev_ss_ts):
				actual_step_end_indices.append(step_end_indices[n+1])

	actual_step_end_indices.append(step_end_indices[-1])

	# makign the dictionary 
	for i in range(len(actual_step_end_indices)):
		step_start = actual_step_start_indices[i]
		step_end = actual_step_end_indices[i]
		avg_time = times[floor((step_start + step_end)/2)]
		avg_temp = np.mean(temps[step_start:step_end])
		errors.append((temps[step_start]-temps[step_end])/2)
		step_mean = {}
		step_mean['Time'] = avg_time
		plot_times.append(avg_time)
		step_mean['Average Temperature'] = avg_temp
		plot_temps.append(avg_temp)
		step_mean_temps.append(step_mean)

	plt.plot(times, temps)
	plt.scatter(plot_times, plot_temps, color='red', label='average time and step temp')
	plt.scatter(times[actual_step_end_indices], temps[actual_step_end_indices], color='purple', label='step ends')
	plt.scatter(times[actual_step_start_indices], temps[actual_step_start_indices], color='green', label='step starts')
	plt.xlabel("Time")
	plt.ylabel("Temperature")
	plt.title("Zoomed temperature steps of module with falling temperature")
	plt.legend()
	plt.xticks(rotation=45)
	plt.grid()
	plt.minorticks_on()
	plt.show()

	if cooling_bool:
		step_start_ends = (fall_idx + np.array(actual_step_start_indices), fall_idx + np.array(actual_step_end_indices))
	else:
		step_start_ends = (rise_idx + np.array(actual_step_start_indices), rise_idx + np.array(actual_step_end_indices))

	return step_mean_temps, np.array(errors), step_start_ends


def find_dt_power_cooling(cu_housing_df, cu_housing_header, plate_df, plate_header, module_df, module_header, 
temps_powers, step_start_ends):	
	"""
	Finds the deltaT versus TEC power (per SiPM array) for the steps in a TEC cooling cycle. The deltaT is calculated 
	with respect to Cu housing and the cold plate (normalised to account for the initial disparity as well). 
	
	There are two options to calculate the dT - 
	a) taking the step temperature as its linearised form and then using the mean of that for the difference
	b) just taking the mean step temperature and using that for the difference
	
	Mean (b) is usually slightly more accurate, and also if we manage to fix the oscillations from the plant then that 
	should be used.  
	
	Arguments:
		cu_housing_df and cu_housing_header - for the Cu housing temperature
		plate_df and plate_header - for the cold plate
		module_df and module_header - for the module
		temps_powers (DF) - output of compare_temp_power_dependencies_2(), which has a column of TEC powers as
							the index and mean temperatures at that power as a column
		step_start_ends (tuple) - step start indices, step end indices
	
	Returns: a DF of dt and powers with module temperature, plate temperature, Cu housing temperature, 
			dT wrt initial module temperature, Cu housing, cold plate, normalised wrt Cu housing and 
			normalised wrt cold plate for each TEC power step. 
		dts_df (DF) - 
			index = powers
			'mod_temps','plate_temps','cu_housing_temps','wrt_mod','wrt_cu','wrt_plate','wrt_cu_normalised','wrt_plate_normalised'
	"""
	start_indices, end_indices = step_start_ends

	# in case there's extraneous steps detected at the end because other changes happened at the end
	if len(start_indices) > len(temps_powers):
		start_indices, end_indices = start_indices[:len(temps_powers)], end_indices[:len(temps_powers)]

	module_fn = module_df[module_header].to_numpy()
	plate_fn = plate_df[plate_header].to_numpy()
	cu_housing_fn = cu_housing_df[cu_housing_header].to_numpy()

	# # to try correcting the oscillations out, we need a longer period of stable oscillations, which all the datasets do not have
	# try: 
	# 	init_temp, error, change_idx = correcting_init_temp(plate_df, plate_header, module_df, module_header, RISE_THRESHOLD, FALL_THRESHOLD, date_str)
	# except:
	rise_idx = find_rise_idx(module_df, module_header, RISE_THRESHOLD)
	fall_idx = find_fall_idx(module_df, module_header, FALL_THRESHOLD)
	change_idx = min(fall_idx, rise_idx)

	init_mod_temps = module_fn[:change_idx]
	init_plate_temps = plate_fn[:change_idx]
	init_cu_temps = cu_housing_fn[:change_idx]

	# the modules are at different initial temperatures than the Cu housing and the cold plate, which
	# shouldn't exist in the ideal case such as in Arjan's setup.
	mod_cu_normalisation = np.mean(init_mod_temps - init_cu_temps)
	mod_plate_normalisation = np.mean(init_mod_temps - init_plate_temps)

	print(f'cu norm: {mod_cu_normalisation}')
	print(f'plate norm: {mod_plate_normalisation}')

	# take the module's initial temperature to be right before it falls due to the TECs
	fall_idx = find_fall_idx(module_df, module_header, FALL_THRESHOLD)
	init_mod_temp = module_fn[fall_idx]

	# rounding the powers
	powers = np.rint(temps_powers.iloc[:,0].to_numpy())
	dts_module_init = [init_mod_temp - module_fn[fall_idx]]
	dts_cu = [init_mod_temp - cu_housing_fn[fall_idx]]
	dts_plate = [init_mod_temp - plate_fn[fall_idx]]
	mod_temps = [init_mod_temp]
	plate_temps = [plate_fn[fall_idx]]
	cu_temps = [cu_housing_fn[fall_idx]]

	for i in range(len(start_indices)):
		step_module_temps = module_fn[start_indices[i]:end_indices[i]]
		step_plate_temps = plate_fn[start_indices[i]:end_indices[i]]
		step_cu_housing_temps = cu_housing_fn[start_indices[i]:end_indices[i]]

		# try:
		# 	linear_temps = [approximate_linear_signal(approximate_signal(temps, DATE_DICT[date_str]['PERIOD_BOUNDS']), temps) for temps in (step_module_temps, step_plate_temps, step_cu_housing_temps)]
		# 	mod_temp = np.mean(linear_temps[0])
		# 	plate_temp = np.mean(linear_temps[1])
		# 	cu_temp = np.mean(linear_temps[2])
		# 	dt_cu = np.mean(linear_temps[0] - linear_temps[2])
		# 	dt_plate = np.mean(linear_temps[0] - linear_temps[1])
		# 	dt_module = np.mean(linear_temps[0] - temps_powers.index.to_numpy()[0])
		# except:

		mod_temp = np.mean(step_module_temps)
		mod_temps.append(mod_temp)

		plate_temp = np.mean(step_plate_temps)
		plate_temps.append(plate_temp)

		cu_temp = np.mean(step_cu_housing_temps)
		cu_temps.append(cu_temp)

		# dt_cu = np.mean(step_module_temps - step_cu_housing_temps)
		dt_cu = mod_temp - cu_temp
		dts_cu.append(dt_cu)

		# dt_plate = np.mean(step_module_temps - step_plate_temps)
		dt_plate = mod_temp - plate_temp
		dts_plate.append(dt_plate)

		# dt_module = np.mean(step_module_temps - temps_powers.index.to_numpy()[0])
		dt_module = mod_temp - init_mod_temp
		dts_module_init.append(dt_module)

		# # To plot the corrected temperature at each step, uncomment this block of code
		# plt.plot(step_module_temps, label='Actual module temp')
		# plt.plot(step_plate_temps, label='Actual plate temp')
		# plt.plot(step_cu_housing_temps, label='Actual Cu housing temp')
		# plt.axhline(y=mod_temp, label='Linear module temp', color='red')
		# plt.axhline(y=plate_temp, label='Linear plate temp', color='black')
		# plt.axhline(y=cu_temp, label='Linear Cu housing temp', color='yellow')
		# plt.plot(module_fn, label='module', color='orange')
		# plt.plot(plate_fn, label='plate', color='blue')
		# plt.plot(cu_housing_fn, label='cu housing', color='green')
		# plt.xlabel('Sample')
		# plt.ylabel('Temperature (ºC)')
		# plt.legend()
		# plt.minorticks_on()
		# plt.show()
	
	normalised_dts_cu = np.array(dts_cu) - mod_cu_normalisation
	normalised_dts_plate = np.array(dts_plate) - mod_plate_normalisation

	fig, axs = plt.subplots(1, 2)

	axs[0].plot(module_df.index.to_numpy(), cu_housing_df[cu_housing_header], label='Cu housing', color='blue')
	axs[0].plot(module_df.index.to_numpy(), module_df[module_header], label='Module', color='orange')
	axs[0].plot(module_df.index.to_numpy(), plate_df[plate_header], label='Cold plate', color='green')
	axs[0].axhline(y=init_mod_temp, label='Initial module temperature', color='red')
	axs[0].set_xlabel('Time')
	axs[0].set_ylabel('Temperature (ºC)')
	axs[0].legend()
	axs[0].set_title('Temperatures over time')
	axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
	axs[0].minorticks_on()

	print(dts_module_init)
	print(powers)

	axs[1].plot(dts_module_init, powers, label='∆T (wrt initial module T)', color='orange', marker='o')
	axs[1].plot(normalised_dts_cu, powers, label='∆T (wrt Cu housing)', color='blue', marker='o')
	axs[1].plot(normalised_dts_plate, powers, label='∆T (wrt cold plate)', color='green', marker='o')
	axs[1].set_xlabel('∆T (ºC)')
	axs[1].set_ylabel('Power (mW)')
	axs[1].set_title('∆T vs TEC Power')
	axs[1].legend()
	axs[1].grid()
	axs[1].minorticks_on()
	plt.show()

	dts_df = pd.DataFrame({
		'mod_temps': mod_temps,
		'plate_temps': plate_temps,
		'cu_housing_temps': cu_temps,
		'wrt_mod': dts_module_init,
		'wrt_cu': dts_cu,
		'wrt_plate': dts_plate,
		'wrt_cu_normalised': normalised_dts_cu,
		'wrt_plate_normalised': normalised_dts_plate
	}, index=powers)


	return round(dts_df, 2)


def find_dt_power_heating(cu_housing_df, cu_housing_header, plate_df, plate_header, module_df, module_header, 
temps_powers, step_start_ends):
	"""
	Same as find_dt_power_cooling() but for the annealing case. 
	
	NOTE: this may have to be re-examined when we have standalone annealing cycles (not taking place after a cooling 
	cycle), where the first detected step isn't the initial temperature, but is the first step. Then it will become
	more like find_dt_power_cooling(). 
	"""
	start_indices, end_indices = step_start_ends

	# in case there's extraneous steps detected at the end because other changes happened at the end
	if len(start_indices) > len(temps_powers):
		start_indices, end_indices = start_indices[:len(temps_powers)], end_indices[:len(temps_powers)]

	module_fn = module_df[module_header].to_numpy()
	plate_fn = plate_df[plate_header].to_numpy()
	cu_housing_fn = cu_housing_df[cu_housing_header].to_numpy()

	# # to try correcting the oscillations out, we need a longer period of stable oscillations, which all the datasets do not have
	# try: 
	# 	init_temp, error, change_idx = correcting_init_temp(plate_df, plate_header, module_df, module_header, RISE_THRESHOLD, FALL_THRESHOLD, date_str)
	# except:
	rise_idx = find_rise_idx(module_df, module_header, RISE_THRESHOLD)
	fall_idx = find_fall_idx(module_df, module_header, FALL_THRESHOLD)
	change_idx = min(fall_idx, rise_idx)

	print(f'change idx: {change_idx}')

	init_mod_temps = module_fn[:change_idx]
	init_plate_temps = plate_fn[:change_idx]
	init_cu_temps = cu_housing_fn[:change_idx]

	# the modules are at different initial temperatures than the Cu housing and the cold plate, which
	# shouldn't exist in the ideal case such as in Arjan's setup.
	mod_cu_normalisation = np.mean(init_mod_temps - init_cu_temps)
	mod_plate_normalisation = np.mean(init_mod_temps - init_plate_temps)

	print(f'cu norm: {mod_cu_normalisation}')
	print(f'plate norm: {mod_plate_normalisation}')

	# for the heating cycle after the cooling cycle, the first step detected is actually the initial temperature
	init_mod_temp = np.mean(module_fn[start_indices[0]:end_indices[0]])
	init_plate_temp = np.mean(plate_fn[start_indices[0]:end_indices[0]])
	init_cu_temp = np.mean(cu_housing_fn[start_indices[0]:end_indices[0]])
	# rounding the powers
	powers = np.rint(temps_powers.iloc[:,0].to_numpy())

	dts_module_init = [init_mod_temp - init_mod_temp]
	dts_cu = [init_mod_temp - init_cu_temp]
	dts_plate = [init_mod_temp - init_plate_temp]
	mod_temps = [init_mod_temp]
	plate_temps = [init_plate_temp]
	cu_temps = [init_cu_temp]

	for i in range(1, len(start_indices)):
		step_module_temps = module_fn[start_indices[i]:end_indices[i]]
		step_plate_temps = plate_fn[start_indices[i]:end_indices[i]]
		step_cu_housing_temps = cu_housing_fn[start_indices[i]:end_indices[i]]

		# try:
		# 	linear_temps = [approximate_linear_signal(approximate_signal(temps, DATE_DICT[date_str]['PERIOD_BOUNDS']), temps) for temps in (step_module_temps, step_plate_temps, step_cu_housing_temps)]
		# 	mod_temp = np.mean(linear_temps[0])
		# 	plate_temp = np.mean(linear_temps[1])
		# 	cu_temp = np.mean(linear_temps[2])
		# 	dt_cu = np.mean(linear_temps[0] - linear_temps[2])
		# 	dt_plate = np.mean(linear_temps[0] - linear_temps[1])
		# 	dt_module = np.mean(linear_temps[0] - temps_powers.index.to_numpy()[0])
		# except:

		mod_temp = np.mean(step_module_temps)
		mod_temps.append(mod_temp)

		plate_temp = np.mean(step_plate_temps)
		plate_temps.append(plate_temp)

		cu_temp = np.mean(step_cu_housing_temps)
		cu_temps.append(cu_temp)

		# dt_cu = np.mean(step_module_temps - step_cu_housing_temps)
		dt_cu = mod_temp - cu_temp
		dts_cu.append(dt_cu)

		# dt_plate = np.mean(step_module_temps - step_plate_temps)
		dt_plate = mod_temp - plate_temp
		dts_plate.append(dt_plate)

		# dt_module = np.mean(step_module_temps - temps_powers.index.to_numpy()[0])
		dt_module = mod_temp - init_mod_temp
		dts_module_init.append(dt_module)

		# # To plot the corrected temperature at each step, uncomment this block of code
		# plt.scatter(step_start_ends[0], module_fn[step_start_ends[0]], label='step starts')
		# plt.scatter(step_start_ends[1], module_fn[step_start_ends[1]], label='step ends')
		# plt.plot(step_module_temps, label='Actual module temp')
		# plt.plot(step_plate_temps, label='Actual plate temp')
		# plt.plot(step_cu_housing_temps, label='Actual Cu housing temp')
		# plt.axhline(y=mod_temp, label='Linear module temp', color='red')
		# plt.axhline(y=plate_temp, label='Linear plate temp', color='black')
		# plt.axhline(y=cu_temp, label='Linear Cu housing temp', color='yellow')
		# plt.plot(module_fn, label='module', color='orange')
		# plt.plot(plate_fn, label='plate', color='blue')
		# plt.plot(cu_housing_fn, label='cu housing', color='green')
		# plt.xlabel('Sample')
		# plt.ylabel('Temperature (ºC)')
		# plt.legend()
		# plt.minorticks_on()
		# plt.show()
	
	normalised_dts_cu = np.array(dts_cu) - mod_cu_normalisation
	normalised_dts_plate = np.array(dts_plate) - mod_plate_normalisation

	dts_df = pd.DataFrame({
		'mod_temps': mod_temps,
		'plate_temps': plate_temps,
		'cu_housing_temps': cu_temps,
		'wrt_mod': dts_module_init,
		'wrt_cu': dts_cu,
		'wrt_plate': dts_plate,
		'wrt_cu_normalised': normalised_dts_cu,
		'wrt_plate_normalised': normalised_dts_plate
	}, index=powers)

	fig, axs = plt.subplots(1, 2)

	axs[0].plot(cu_housing_df[cu_housing_header], label='Cu housing', color='blue')
	axs[0].plot(module_df[module_header], label='Module', color='orange')
	axs[0].plot(plate_df[plate_header], label='Cold plate', color='green')
	axs[0].axhline(y=init_mod_temp, label='Initial module temperature', color='red')
	axs[0].set_xlabel('Time')
	axs[0].set_ylabel('Temperature (ºC)')
	axs[0].legend()
	axs[0].set_title('Temperatures over time')
	axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation = 45)
	axs[0].minorticks_on()


	axs[1].plot(dts_module_init, powers, label='∆T (wrt initial module T)', color='orange', marker='o')
	axs[1].plot(normalised_dts_cu, powers, label='∆T (wrt Cu housing)', color='blue', marker='o')
	axs[1].plot(normalised_dts_plate, powers, label='∆T (wrt cold plate)', color='green', marker='o')
	axs[1].set_xlabel('∆T (ºC)')
	axs[1].set_ylabel('Power (mW)')
	axs[1].set_title('∆T vs TEC Power')
	axs[1].legend()
	axs[1].grid()
	axs[1].minorticks_on()
	plt.show()

	return round(dts_df, 2)


def measure_second_correlation(sig, sig_header, sig_noise, sig_noise_header, step_start_indices, step_end_indices):
	"""
	DEFUNCT
	when the heaters are turned on, i.e., plant temperature becomes less than the tray temperature, 
	find a different correlation & correction
	assumes the dfs are reformatted already, and start at the same point
	"""
	# corrections = []
	# for i in range(len(step_start_indices)):
	# 	step_sig_df = sig.iloc[step_start_indices[i]:step_end_indices[i]]
	# 	# print(step_sig_df)
	# 	corr_devs, devs, translation = correcting_temp(sig_noise, sig_noise_header, step_sig_df, sig_header)
	# 	print(f"corr devs, devs: {corr_devs, devs}")
	# 	plt.plot(step_sig_df, label='step temperature', color='purple')
	# 	plt.axhline(y=(corr_devs+translation), label='corr devs', color='blue')
	# 	plt.axhline(y=translation, label='translation', color='red')
	# 	plt.axhline(y=(devs+translation), label='devs', color='green')
	# 	plt.xticks(rotation=45)
	# 	plt.legend()
	# 	plt.minorticks_on()
	# 	plt.show()
	# 	percent_correction = abs(corr_devs - devs)/devs * 100
	# 	corrections.append({'Step':i, 'Corr devs':corr_devs, 'Devs':devs, 'Improvement%':percent_correction, 'Translation':translation})
	# corrections = pd.DataFrame(corrections)
	# return corrections
	corrections = []
	# fig, axs = plt.subplots(1, len(step_start_indices))
	for i in range(len(step_start_indices)):
		step_sig_df = sig.iloc[step_start_indices[i]:step_end_indices[i]]
		corr_devs, devs, translation = correcting_temp(sig_noise, sig_noise_header, step_sig_df, sig_header)
		# axs[i].plot(step_sig_df, label='step temperature', color='purple')
		# axs[i].axhline(y=(corr_devs+translation), label='corr devs', color='blue')
		# axs[i].axhline(y=translation, label='translation', color='red')
		# axs[i].axhline(y=(devs+translation), label='devs', color='green')
		# for tick in axs[i].get_xticklabels():
		# 	tick.set_rotation(45)
		# axs[i].set_title(f'Step {i+1}')
		percent_correction = abs(corr_devs - devs)/devs * 100
		corrections.append({'Step':i, 'Corr devs':corr_devs, 'Devs':devs, 'Improvement%':percent_correction, 'Translation':translation})
	# plt.legend()
	# plt.xticks(rotation=45)
	# plt.minorticks_on()
	# plt.show()

	corrections = pd.DataFrame(corrections)
	return corrections


def exponential_fit(xs, a, b, c):	
	"""
	Defines an exponential decay fit with equation y = a * e ^{-b * xs} + c
	"""
	return a * np.exp(-b * xs) + c


def fit_temps_powers(temps_powers, errors):
	"""
	Fits the temperature vs TEC power graph to different decay functions. 
	"""
	xs, ys = temps_powers
	xs, ys = np.array(xs), np.array(ys)
	# EXPONENTIAL DECAY
	popt, _ = curve_fit(exponential_fit, xs, ys, sigma=errors,absolute_sigma=True, maxfev=2000)
	a_opt, b_opt, c_opt = popt
	y_fit = exponential_fit(xs, a_opt, b_opt, c_opt)
	print(f"r^2 value: {r2_score(ys, y_fit)}")

	print(f"Exponential decay function: y = {a_opt}exp(-{b_opt}x) + {c_opt}")
	print(f"errors: {errors}")

	# RATIONAL DECAY
	# popt, _ = curve_fit(rational_decay, xs, ys)
	# a_opt, b_opt, c_opt = popt
	# y_fit = rational_decay(xs, a_opt, b_opt, c_opt)
	# print(f"Rational decay function: y = {a_opt}/x + {b_opt}")
	
	# SPLINES
	# Fit a spline to the data
	# spl = UnivariateSpline(xs, ys)
	# y_fit = spl(xs)
	# coeffs = spl.get_coeffs()
	# knots = spl.get_knots()
	# print(fitted_function(xs, coeffs, knots))
	# y_fit = fitted_function(xs, coeffs, knots)

	# LINEAR
	plt.scatter(xs, ys, label='Data')
	plt.plot(xs, y_fit, 'r', label='Fitted Curve')
	plt.title("Functional fit of Power with Temperature")
	plt.xlabel('Temperature (ºC)')
	plt.ylabel('Power (mW)')
	plt.legend()
	plt.minorticks_on()
	plt.show()


def convert_datetime_to_onlytime(datetime_lst):
	"""
	DEFUNCT 
	Converts datetime formats to only time (removes date), returns a list of dictionaries with key 'Time'
	"""
	fixed_lst = []
	for i in range(len(datetime_lst)):
		time = (datetime_lst[i])['Time']
		words = time.split()
		req_time = words[-1]
		other_column = list(datetime_lst[i].keys())[1]
		fixed_lst.append({'Time': req_time, other_column: (datetime_lst[i])[other_column]})

	print(f"Datetime conversion: {fixed_lst}")
	return fixed_lst


def compare_temp_power_dependencies(plate_df, plate_header, plant_df, plant_header, module_df, module_header, supply_df, supply_header, elog_filename):
	"""
	DEFUNCT - use compare_temp_power_dependencies_2() instead. 
	Compares the module temperature vs TEC power
	"""
	time_powers, not_interpolated_time_powers = read_power_from_elog(elog_filename)
	# time_temps, errors, (starts, ends) = find_step_avg_temp(plate_df, plate_header, module_df, module_header, supply_df, supply_header)
	# print(f"Time powers: {time_powers}")
	# print(f"Time temps: {time_temps}")

	time_temps, errors, (starts, ends) = find_step_avg_temp2(module_df, module_header, FALL_THRESHOLD, RISE_THRESHOLD)

	# init_temp, init_temp_error, change_idx = correcting_init_temp(plate_df, plate_header, module_df, module_header, PLATE_PLANT_THRESHOLD, MODULE_PLANT_THRESHOLD)
	# new list of temperatures and powers
	# temps = [init_temp]
	# powers = [0]

	temps = []
	powers = []

	# get closest time_power to the time_temp and then compare them - if they have the same minute
	for i in range(len(time_temps)):
		# time_stamp_power = (time_powers.index)[i]
		# for j in range(len(time_temps)):
		# 	time_stamp_temp_1 = (time_temps[j])['Time']
		# 	# find the closest time 
		# 	# if (time_stamp_power >= time_stamp_temp_1) and (time_stamp_power <= time_stamp_temp_2):
		# 	if (time_stamp_power == time_stamp_temp_1):
		# 		dt = (time_temps[j])['Average Temperature'] 
		# 		temps.append(dt)
		# 		powers.append((time_powers['Power'])[i])
		# 	# elif (i == len(time_powers) - 1):
		# 	# 	dt = (time_temps[j])['Average Temperature'] 
		# 	# 	temps.append(dt)
		# 	# 	powers.append((time_powers['Power'])[-1])
		temps.append((time_temps[i])['Average Temperature'])
		powers.append(not_interpolated_time_powers['Power'][i])
	
	temps_powers = (temps, powers)
	
	temp_power_df = pd.DataFrame(temps_powers)
	print(temp_power_df)
	return temps_powers, errors


def compare_temp_power_dependencies_2(module_df, module_header, elog_filename, cooling_bool):
	"""
	Compares the module temperature to the TEC power at that step. 
	a) for standalone heating cycles, because the initial temperature is also detected as a step and so we currently discount
	that - if you want to count the first detection as a step, comment block 1 
	b) if other tests are done (like adding/removing insulation), which introduces inconsistencies in the 
		temperature similar to a step, then there may be trailing steps detected, so comment the second 
		block of code if you don't want to account for that. 
	
	Arguments:
		module_df and module_header - for the module temperature
		elog_filename (str) - elog for that day's powers
		cooling_bool (bool) - True if cooling cycle, False if heating cycle
	
	Returns: 
		temp_power_df (DF) - DataFrame with power as index and avg temperature at that TEC step as column, 
							rounded to 2 decimal points
		errors (array) - error bars associated with each step temperature
	"""
	time_powers, not_interpolated_time_powers = read_power_from_elog(elog_filename)
	time_temps, errors, (starts, ends) = find_step_avg_temp2(module_df, module_header, FALL_THRESHOLD, RISE_THRESHOLD, cooling_bool)

	# for all the heating cycles - because it detects the return to initial temperature also as a step
	if not cooling_bool:
		time_temps = time_temps[1:]

	temps = []
	powers = []

	# in case of extra steps at the end when there's subtle changes like adding/removing insulation
	if len(time_temps) > len(not_interpolated_time_powers):
		time_temps = time_temps[:len(not_interpolated_time_powers)]

	for i in range(len(time_temps)):
		temps.append((time_temps[i])['Average Temperature'])
		powers.append(np.rint(not_interpolated_time_powers['Power'][i]))
	
	temp_power_df = pd.DataFrame(powers, temps)
	print(round(temp_power_df, 2))
	return round(temp_power_df, 2), errors


def plot_temp_power_dependencies(temps_powers):	
	"""
	Plots the step temperature vs power and dT vs power. 
	Uncomment the block of code to get a smoother curve with a spline. 
	
	Arguments: temps_powers (DF) - power as index and temperature as column
	"""
	fig, axs = plt.subplots(1, 2)
	x_axis, y_axis = temps_powers.index.to_numpy(), temps_powers.iloc[:,0].to_numpy()

	# x_new = np.linspace(x_axis.min(), x_axis.max(), 20)
	# spl = make_interp_spline(x_axis, y_axis, k=3)
	# y_axis_smooth = spl(x_new)	
	# axs[0].plot(x_new, y_axis_smooth, color='green')

	axs[0].plot(x_axis, y_axis, color='green', marker='o')
	axs[0].set_title('Temperature of Module vs Power')
	axs[0].set_xlabel('Temperature (ºC)')
	axs[0].set_ylabel('TEC Power (mW)')
	axs[0].grid()
	axs[0].minorticks_on()
	
	dT = np.array([x_axis[i] - x_axis[0] for i in range(len(x_axis))])
	axs[1].plot(dT, y_axis, color='red', marker='o')
	axs[1].set_title('dT of Module vs Power')
	axs[1].set_xlabel('dT (ºC)')
	axs[1].set_ylabel('TEC Power (mW)')
	axs[1].grid()
	axs[1].minorticks_on()
	plt.show()


def compare_rms_with_window_size(noisy_filename1, noisy_header1, signal_filename2, signal_header2):	
	"""
	DEFUNCT
	Comparing rms of deviations with window size to figure out best window size for rolling means. 
	"""
	noise = reformat_df(noisy_filename1, noisy_header1)
	signal = reformat_df(signal_filename2, signal_header2)
	cut_noise, cut_signal = start_end_same_timestamp(noise, signal)

	sig = cut_signal[signal_header2]
	sig_noise = cut_noise[noisy_header1]
	
	windows = []
	rms_noises = []
	rms_signals = []

	for window in range(100):
		noise_moving_averages = find_moving_average(cut_noise, noisy_header1, f'{window}s')
		noise_deviations = sig_noise - noise_moving_averages
		rms_noise = np.max(noise_deviations)/sqrt(2)
		signal_moving_averages = find_moving_average(cut_signal, signal_header2, f'{window}s')
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
	plt.minorticks_on()
	plt.show()


def measure_phase_shift(noise_popt, signal_popt):
	"""
	Measure the phase shift between a noise function and a signal function. 
	
	Arguments:
		noise_popt and signal_popt - (amp, period, translation, phi, m, k = popt) for the noise and signal
	
	Returns:
		phase_shift - phase shift between noise and signal sinusoidal functions in seconds 
	"""
	x = np.arange(len(noise_popt))
	n_amp, n_period, n_translation, n_phi, n_m, n_k = noise_popt
	noise_fn = n_amp * np.sin(2 * np.pi/n_period * (x - n_phi)) + n_translation + n_m * x + n_k

	s_amp, s_period, s_translation, s_phi, s_m, s_k = signal_popt
	signal_fn = s_amp * np.sin(2 * np.pi/s_period * (x - s_phi)) + s_translation + s_m * x + s_k

	cross_corr = correlate(signal_fn, noise_fn)
	lag = np.argmax(cross_corr)
	lags = np.arange(1-len(signal_fn), len(noise_fn))

	# each sample is every 10 seconds, so when finding lag of samples in seconds, multiply by 10
	time_step = 10 	
	lag_in_seconds = lags[lag] * time_step
	phase_shift = (s_phi - n_phi) * time_step

	# fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
	# ax_orig.plot(signal_fn)
	# ax_orig.set_title('Original signal')
	# ax_orig.set_xlabel('Sample Number')
	# ax_noise.plot(noise_fn)
	# ax_noise.set_title('Signal with noise')
	# ax_noise.set_xlabel('Sample Number')
	# ax_corr.plot(lags, cross_corr)
	# ax_corr.set_title('Cross-correlated signal')
	# ax_corr.set_xlabel('Lag')
	# ax_orig.margins(0, 0.1)
	# ax_noise.margins(0, 0.1)
	# ax_corr.margins(0, 0.1)
	# fig.tight_layout()
	# plt.minorticks_on()
	# plt.show()

	# plt.figure()
	# plt.plot(lags, cross_corr)
	# plt.xlabel("Lag")
	# plt.ylabel("Cross-correlation")
	# plt.title("Cross-correlation between signal_fn and noise_fn")
	# plt.minorticks_on()
	# plt.show()

	return phase_shift

	
def measure_cross_correlation(noise_df, noise_header, signal_df, signal_header):
	"""
	DEFUNCT
	Measures the cross-correlation of two functions in a flat period (such as the initial temperature).
	"""
	noise_period, noise_peak_indices, noise_trough_indices = find_oscillation_freq(noise_df, noise_header)
	if find_oscillation_freq(signal_df, signal_header) != []:
		signal_period, signal_peak_indices, signal_trough_indices = find_oscillation_freq(signal_df, signal_header)
		
		noise_fn = noise_df[noise_header].to_numpy()
		signal_fn = signal_df[signal_header].to_numpy()

		cross_corr = correlate(noise_fn, signal_fn)
		lag = np.argmax(cross_corr)
		# lags = correlation_lags(len(signal_fn), len(noise_fn))

		# scaling_factor = np.max(noise_fn[noise_peak_indices]) / np.max(signal_fn[signal_peak_indices])
		noise_amp = (np.mean(noise_fn[noise_peak_indices]) - np.mean(noise_fn[noise_trough_indices]))/2
		signal_amp = (np.mean(signal_fn[signal_peak_indices]) - np.mean(signal_fn[signal_trough_indices]))/2
		# print(f"signal amp: {signal_amp}")
		scaling_factor = noise_amp/signal_amp

# TO DEMONSTRATE THE SCALING OF MODULE TO PLATE
	# plt.plot(noise_fn, label='plate temperature')
	# plt.plot(signal_fn, label='module temperature')
	# plt.plot(((signal_fn + 32) * scaling_factor) - 33.1, label='module * scaling factor')
	# plt.plot((noise_fn + 33.1)/scaling_factor - 32, label='plate/scaling factor')
	# plt.legend()
	# plt.minorticks_on()
	# plt.show()
	
# TO PLOT THE CROSS-CORRELAITON 
	# fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
	# ax_orig.plot(signal_fn)
	# ax_orig.set_title('Original signal')
	# ax_orig.set_xlabel('Sample Number')
	# ax_noise.plot(noise_fn)
	# ax_noise.set_title('Signal with noise')
	# ax_noise.set_xlabel('Sample Number')
	# ax_corr.plot(lags, cross_corr)
	# ax_corr.set_title('Cross-correlated signal')
	# ax_corr.set_xlabel('Lag')
	# ax_orig.margins(0, 0.1)
	# ax_noise.margins(0, 0.1)
	# ax_corr.margins(0, 0.1)
	# fig.tight_layout()
	# plt.minorticks_on()
	# plt.show()

		return lag, scaling_factor


def linear_fit(x, a, b):	
	"""
	Defines a linear function y = ax+b
	"""
	return a*x + b


def parabolic_fit(x, a, b, c):	
	"""
	Defines a parabolic function y = ax^2 + bx + c
	"""
	return a*(x**2) + b*x + c



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~THEORY SECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

R_AC_VALUES = [3.73, 2.58] # Ohms
R_AC_ERRORS = [0.07, 0.05]
ALPHA_VALUES = [14.7, 12.3] # mV/K
ALPHA_ERRORS = [0.3, 0.2]
K_VALUES = [22.2, 23.3] # mW/K
K_ERRORS = [0.4, 0.5]


def finding_coefficients(at_25_and_neg35, errors):
	"""
	Used to find the values of TEC resistance with alternating current (R_ac), the Seebeck coefficient (alpha),
	and conductance (K) given each of their paper values at 25 ºC and -35 ºC.
	
	Arguments:
		at_25_and_neg35 (tuple) - value of coefficient at  25 ºC and -35 ºC
		errors (tuple) - uncertainties associated with those values
	
	Returns:
		a_opt (float) - sloope of linear fit
		b_opt (float) - translation of linear fit
	"""
	temperatures = np.array([25, -35])
	popt, _ = curve_fit(linear_fit, temperatures, np.array(at_25_and_neg35), sigma=errors, absolute_sigma=True)
	a_opt, b_opt = popt
	y_fit = linear_fit(np.array(at_25_and_neg35), a_opt, b_opt)
	plt.plot(temperatures, at_25_and_neg35)
	plt.xlabel('Temperature')
	plt.grid()
	plt.minorticks_on()
	plt.show()
	print(f'equation: y = {a_opt}x + {b_opt}')
	return a_opt, b_opt


def find_linear_R_ac(T):
	"""
	Finding the TEC resistance at a given temperature using the linear fit found with finding_coefficients().
	Arguments: T (float) - temperature
	Returns: approximate R_ac value at that temperature (in Ohms)
	"""
	return (0.01916666666666666 * T + 3.2508333333333335)


def find_linear_K(T):	
	"""
	Finding the TEC conductance at a given temperature using the linear fit found with finding_coefficients().
	Arguments: T (float) - temperature
	Returns: approximate K value at that temperature (in mW/K)
	"""
	return (-0.018333333333333326 * T + 22.65833333333333)


def find_linear_alpha(T):
	"""
	Finding the Seebeck coefficient at a given temperature using the linear fit found with finding_coefficients().
	Arguments: T (float) - temperature
	Returns: approximate alpha value at that temperature (in mV/K)
	"""
	return (0.03999999999999998 * T + 13.700000000000001)


def paper_quad_alpha(T):	
	"""
	Formula found in FEM thermoelectric modelling for bismuth telluride Seebeck coefficient 
		- but doesn't work for the TECs as a whole.
	"""
	return (1.802 * 10**(-4) + 3.861 * 10**(-7) * T - 9.582 * 10**(-10) * T**2) * 1000


def compare_linear_and_quad_alpha():
	"""
	Compares the Seebeck coefficients of TECs (from linear model) and just bismuth telluride (from quadratic model).
	Significant difference between linear and quadratic found. 
	"""
	ts = np.arange(-50, 50)
	linear_alphas = find_linear_alpha(ts)
	print(f'lin: {linear_alphas}')
	# but the quadratic coefficients would change if using K instead of ºC, which we do not have
	quad_alphas = paper_quad_alpha(ts + 273.15)
	print(f'quad: {quad_alphas}')
	plt.plot(ts, linear_alphas, label='linear alphas')
	plt.plot(ts, quad_alphas, label='quadratic alphas')
	plt.xlabel('Temperature (ºC)')
	plt.ylabel('Seebeck coefficient (mV/K)')
	plt.legend()
	plt.minorticks_on()
	plt.show()


def paper_predicted_k(T):
	"""
	Use Peltier equation to predict K values of bismuth telluride:
		k(T) = 1.758 - 5.290 * 10^13 T + 4.134 * 10 ^-5 T^2
	Returns: K(T) = kA/L
		- Doesn't work for entire TEC, only for bismuth telluride.
	"""
	k = 1.758 - 5.290 * 10**(13) * T + 4.134 * 10**(-5) * T**2
	A = 0.003 * 0.004 	# m^2
	L = 0.001			# m
	K = k * A / L
	return K


def voltage_fit(alpha, dT, R_ac, I):
	"""
	Returns the voltage in the TECs according to the formula V = alpha * dT + I * R_ac
	"""
	return (alpha * dT) + (R_ac * I)


def qc_fit(alpha, I, T_c, R_ac, K_fit, dT):
	"""
	Returns the SiPM power (Qc) according to the formula:
		Qc = (alpha * T_c * I) - (0.5 * (I ** 2) * R_ac) - (K_fit * dT)
	We consider K_fit = K_ev + K
	"""
	return (alpha * T_c * I) - (0.5 * (I ** 2) * R_ac) - (K_fit * dT)


def predicted_qc_values(lyso_df, lyso_header, h, t_inf, step_start_ends):
	"""
	Returns the Qc values that is derived thermally with the equation:
		qc = h(ts - tinf) * 0.5 * surface area of LYSO crystals * no. of crystals per package
			[h is the heat transfer coefficient]
	
	Arguments:
		t_inf (array of floats) - ambient temperature over the period being considered
		lyso_df and lyso_header - for LYSO temperature
		h (int/float) - heat transfer coefficient
	
	Returns:
		array of floats - Qc values in the time period of the DataFrame
	"""
	# h = 10 			# heat transfer coefficient
	sa = 0.058 * 0.003 		# one surface area of LYSO crystal
	n = 16					# number of crystals per array
	
	print(step_start_ends)

	lyso_temp = lyso_df[lyso_header].to_numpy()
	t_s = np.array([np.mean(lyso_temp[step_start_ends[0][i]:step_start_ends[1][i]]) for i in range(len(step_start_ends[0]))])
	t_inf = np.array([np.mean(t_inf[step_start_ends[0][i]:step_start_ends[1][i]]) for i in range(len(step_start_ends[0]))])

	print(f'ts: {t_s}')
	print(f't_inf: {t_inf}')

	qcs = h * (t_s - t_inf) * 0.5 * n * sa

	print(f'qc = {h * 0.5 * n * sa}*(T-t_inf)')

	# plt.plot(t_s, qcs, label='Qc values varying with module temperature', marker='o')
	# plt.title('Q_ambient values')
	# plt.xlabel('Module temperature (ºC)')
	# plt.ylabel('Q_ambient (W)')
	# plt.minorticks_on()
	# plt.show()

	return qcs


def predicted_voltage_fit(dts_df, elog_filename):
	"""
	Compares actual voltage to predicted voltage according to theory.
	
	Arguments:
		dts_df (DataFrame) - output of find_dt_power()
		elog_filename (str) - elog CSV file
	
	Returns:
		voltage_df (DF) - predicted voltages, actual voltages, absolute errors, percentage errors
			'pred_vs','actual_vs','abs_error','percent_error'
	"""
	# because of the voltage being made of the thermovoltage and electric voltage, we know the thermovoltage
	# must be positive (real) so we need the dts to be positive
	dts = dts_df['wrt_cu_normalised'].to_numpy() 
	# dts = dts_df['wrt_plate_normalised'].to_numpy()

	currents = np.array(read_current_from_elog(elog_filename)) / 16

	t_mod = dts_df['mod_temps'].to_numpy()
	t =	dts_df['cu_housing_temps'].to_numpy()
	# t = dts_df['plate_temps'].to_numpy()
	t_bar = (t_mod + t)/2

	# we multiply the coefficients by 4 because there are 4 TECs in series on each array
	alpha = find_linear_alpha(t_bar) * 0.001 * 4
	r_ac = find_linear_R_ac(t_bar) * 4

	# trying to just use -35 ºC values as an approximation - 
		# but not a better approximation than linearly extrapolated values
	# r_ac_cu = 2.58
	# alpha_cu = 0.0123

	pred_vs = voltage_fit(alpha, dts, r_ac, currents) 
	actual_vs = np.array(read_voltage_from_elog(elog_filename))

	plt.plot(dts, pred_vs, label='Predicted voltage (Cu housing)')
	plt.plot(dts, actual_vs, label='Actual voltage')
	plt.title('Voltage compared to ∆T (wrt Cu housing)')
	plt.legend()
	plt.xlabel('∆T (ºC)')
	plt.ylabel('Voltage (V)')
	plt.grid()
	plt.minorticks_on()
	plt.show()

	abs_error = pred_vs - actual_vs
	percent_error = 100 * (pred_vs - actual_vs)/actual_vs

	voltage_df = pd.DataFrame({
		'pred_vs': pred_vs, 
		'actual_vs': actual_vs,
		'abs_error': abs_error,
		'percent_error': percent_error
	})

	return voltage_df


def predicted_rac_values(dts_df, elog_filename, cu_bool):	
	"""
	Compares actual voltage to predicted voltage according to theory. dT can be calculated wrt Cu housing
	or cold plate.
	
	Arguments:
		dts_df (DataFrame) - output of find_dt_power()
		elog_filename (str) - elog CSV file
		cu_bool (bool) - if true, calculates dt wrt cu; else calculates dt wrt plate
	
	Returns:
		rac_df (DF) - predicted R_ac, actual R_ac, absolute errors, percentage errors
			'pred_rac','actual_rac','abs_error','percent_error'
	"""
	if cu_bool:
		dts = dts_df['wrt_cu_normalised'].to_numpy()
		t =	dts_df['cu_housing_temps'].to_numpy()
	else:
		dts = dts_df['wrt_plate_normalised'].to_numpy()
		t = dts_df['plate_temps'].to_numpy()

	currents = np.array(read_current_from_elog(elog_filename)) / (ELOG_MODULES[elog_filename] * 4)

	t_mod = dts_df['mod_temps'].to_numpy()
	
	t_bar = (t_mod + t)/2
	alpha = find_linear_alpha(t_bar) * 0.001 * 4
	
	actual_vs = np.array(read_voltage_from_elog(elog_filename))

	pred_rac = find_linear_R_ac(t_bar) * 4
	actual_rac = (actual_vs - alpha * abs(dts))/currents
	
	# plt.plot(dts, pred_rac, label='Predicted R_ac')
	# plt.plot(dts, actual_rac, label='Actual R_ac')
	# plt.title('Predicted R_ac compared to observed R_ac')
	# plt.legend()
	# plt.xlabel('∆T (ºC)')
	# plt.ylabel('R_ac (ohms)')
	# plt.grid()
	# plt.minorticks_on()
	# plt.show()

	abs_error = pred_rac - actual_rac
	percent_error = 100 * (pred_rac - actual_rac)/actual_rac

	rac_df = pd.DataFrame({
		'pred_rac': pred_rac, 
		'actual_rac': actual_rac,
		'abs_error': abs_error,
		'percent_error': percent_error
	})

	return rac_df 


def predicted_kfit(dts_df, elog_filename, qc=0.480, rac_cu=None, rac_plate=None):	
	"""
	Predicts the K_fit (= K + K_ev) value with the formula:
		K_fit = [(alpha * I * T_c) - (0.5 * R_ac * I^2) - Q_c]/∆T
	
	Arguments:
		dts_df (DataFrame) - output of find_dt_power()
		elog_filename (str) - elog CSV file
		qc (float) [optional] - SiPM load in Watts; default value 480 mW/SiPM array
		rac_cu (float) [optional] - TEC resistance using dT wrt Cu housing; default uses find_linear_R_ac()
		rac_plate (float) [optional] - TEC resistance using dT wrt cold plate; default uses find_linear_R_ac()
	
	Returns:
		k_fit_df (DF) - 'k_fit_wrt_cu', 'k_fit_wrt_plate'
	"""
	dts_wrt_cu = dts_df['wrt_cu_normalised'].to_numpy()
	dts_wrt_plate = dts_df['wrt_plate_normalised'].to_numpy()

	power = dts_df.index.to_numpy()
	t_mod = dts_df['mod_temps'].to_numpy()
	t_cu =	dts_df['cu_housing_temps'].to_numpy()
	t_plate = dts_df['plate_temps'].to_numpy()

	t_bar_wrt_cu = (t_mod + t_cu)/2
	t_bar_wrt_plate = (t_mod + t_plate)/2

	alpha_cu = find_linear_alpha(t_bar_wrt_cu) * 0.001 * 4
	alpha_plate = find_linear_alpha(t_bar_wrt_plate) * 0.001 * 4
	if rac_cu is None:
		r_ac_cu = find_linear_R_ac(t_bar_wrt_cu) * 4
	else:
		r_ac_cu = rac_cu

	if rac_plate is None:
		r_ac_plate = find_linear_R_ac(t_bar_wrt_plate) * 4
	else:
		r_ac_plate = rac_plate

	# alpha_cu = 0.0123
	# r_ac_cu = 2.58
	# alpha_plate = 0.0123
	# r_ac_plate = 2.58

	currents = np.array(read_current_from_elog(elog_filename))/16

	k_fit_paper = (23.3 + 63) * 10**(-3)

	# paper formula
	k_fit_cu = ((alpha_cu * currents * t_bar_wrt_cu) - (0.5 * r_ac_cu * (currents ** 2)) - qc)/(dts_wrt_cu)
	k_fit_plate = ((alpha_plate * currents * t_bar_wrt_plate) - (0.5 * r_ac_plate * (currents ** 2)) - qc)/(dts_wrt_plate)

	fig, axs = plt.subplots(1, 2)
	axs[0].plot(-1 * dts_wrt_plate, k_fit_plate, label='wrt plate', marker='o')
	axs[0].plot(-1 * dts_wrt_cu, k_fit_cu, label='wrt Cu housing', marker='s')
	axs[0].axhline(y=k_fit_paper, label='Paper prediction', color='black')
	axs[0].set_title('K_fit vs ∆T')
	axs[0].set_ylabel('K_fit (K + K_ev) (W/K)')
	axs[0].set_xlabel('∆T (K)')
	axs[0].legend()
	axs[0].grid()
	axs[0].minorticks_on()

	axs[1].plot(k_fit_plate, power, label='wrt plate', marker='o')
	axs[1].plot(k_fit_cu, power, label='wrt Cu housing', marker='s')
	axs[1].axvline(x=k_fit_paper, label='Paper prediction', color='black')
	axs[1].set_xlabel('K_fit (K + K_ev) (W/K)')
	axs[1].set_ylabel('Power (mW)')
	axs[1].set_title('TEC Power vs K_fit')
	axs[1].legend()
	axs[1].grid()
	axs[1].minorticks_on()
	plt.show()


	k_fit_df = pd.DataFrame({
		'k_fit_wrt_cu': k_fit_cu,
		'k_fit_wrt_plate': k_fit_plate
	}, index=power)

	return round(k_fit_df, 2)


def better_predicted_kfit(dts_df, elog_filename, lyso_df, lyso_header, t_inf_df, t_inf_header, step_start_ends, qc=-0.480, rac_cu=None, rac_plate=None, q_elec=-0.806):
	"""
	More comprehensive Q_c used where Q_c = Q_SiPM + Q_ambient + Q_electronics
	Q_SiPM is power injected into each SiPM array = total SiPM power/4 * num_modules
	Q_ambient is calculated as the thermoelectric power derived from the temperature gradient between the LYSO and environment
	Q_electronics is FE power per SiPM array = (TOFHIRs + ALDOs powers)/4 * num_modules  
	
	Otherwise works with same formula as predicted_kfit(). 
	
	Compares the calculated K, K_ev, and K_fit values with Arjan's paper values. 
	"""
	dts_wrt_cu = dts_df['wrt_cu_normalised'].to_numpy()[1:]
	dts_wrt_plate = dts_df['wrt_plate_normalised'].to_numpy()[1:]

	power = dts_df.index.to_numpy()[1:]
	t_mod = dts_df['mod_temps'].to_numpy()[1:]
	t_cu =	dts_df['cu_housing_temps'].to_numpy()[1:]
	t_plate = dts_df['plate_temps'].to_numpy()[1:]

	t_bar_wrt_cu = (t_mod + t_cu)/2
	t_bar_wrt_plate = (t_mod + t_plate)/2

	alpha_cu = find_linear_alpha(t_bar_wrt_cu) * 0.001 * 4
	alpha_plate = find_linear_alpha(t_bar_wrt_plate) * 0.001 * 4
	if rac_cu is None:
		r_ac_cu = find_linear_R_ac(t_bar_wrt_cu) * 4
	else:
		r_ac_cu = rac_cu[1:]

	if rac_plate is None:
		r_ac_plate = find_linear_R_ac(t_bar_wrt_plate) * 4
	else:
		r_ac_plate = rac_plate[1:]

	k_fit_paper = (23.3 + 63) * 10**(-3)

	pred_k_cu = find_linear_K(t_bar_wrt_cu) * 0.001
	pred_k_plate = find_linear_K(t_bar_wrt_plate) * 0.001

	currents = (np.array(read_current_from_elog(elog_filename))/16)[1:]

	h = 10	# heat transfer coefficient approximation for something between free and forced air 
	t_inf = t_inf_df[t_inf_header].to_numpy()

	q_amb = predicted_qc_values(lyso_df, lyso_header, h, t_inf, step_start_ends)
	actual_qc = qc + q_elec + q_amb

	k_fit_cu = ((alpha_cu * currents * t_bar_wrt_cu) - (0.5 * r_ac_cu * (currents ** 2)) - actual_qc)/abs(dts_wrt_cu)
	k_fit_plate = ((alpha_plate * currents * t_bar_wrt_plate) - (0.5 * r_ac_plate * (currents ** 2)) - actual_qc)/abs(dts_wrt_plate)

	k_ev_cu = k_fit_cu - pred_k_cu
	k_ev_plate = k_fit_plate - pred_k_plate


	fig_1, axs_1 = plt.subplots(1, 2)
	axs_1[0].plot(-1 * dts_wrt_plate, k_fit_plate, label='wrt plate', marker='o')
	axs_1[0].plot(-1 * dts_wrt_cu, k_fit_cu, label='wrt Cu housing', marker='s')
	axs_1[0].scatter(10, 0.0863, color='green', marker='P', label='Paper measurement')
	axs_1[0].set_title('K_fit vs ∆T')
	axs_1[0].set_ylabel('K_fit (K + K_ev) (W/K)')
	axs_1[0].set_xlabel('∆T (K)')
	axs_1[0].legend()
	axs_1[0].grid()
	axs_1[0].minorticks_on()

	axs_1[1].plot(k_fit_plate, power, label='wrt plate', marker='o')
	axs_1[1].plot(k_fit_cu, power, label='wrt Cu housing', marker='s')
	axs_1[1].scatter(0.0863, 420, color='green', marker='P', label='Paper measurement')
	axs_1[1].set_xlabel('K_fit (K + K_ev) (W/K)')
	axs_1[1].set_ylabel('Power (mW)')
	axs_1[1].set_title('TEC Power vs K_fit')
	axs_1[1].legend()
	axs_1[1].grid()
	axs_1[1].minorticks_on()
	plt.show()

	fig_2, axs_2 = plt.subplots(1, 2)

	axs_2[0].plot(-1 * dts_wrt_plate, k_ev_plate, label='K_ev wrt plate', marker='o')
	axs_2[0].plot(-1 * dts_wrt_cu, k_ev_cu, label='K_ev wrt Cu housing', marker='s')
	axs_2[0].scatter(10, 0.063, color='green', marker='P', label='K_ev Paper measurement')
	axs_2[0].set_title('K_ev vs ∆T')
	axs_2[0].set_ylabel('K_ev (W/K)')
	axs_2[0].set_xlabel('∆T (K)')
	axs_2[0].legend()
	axs_2[0].grid()
	axs_2[0].minorticks_on()

	axs_2[1].plot(k_ev_plate, power, label='K_ev wrt plate', marker='o')
	axs_2[1].plot(k_ev_cu, power, label='K_ev wrt Cu housing', marker='s')
	axs_2[1].scatter(0.063, 420, color='green', marker='P', label='K_ev paper measurement')
	axs_2[1].set_xlabel('K_ev (W/K)')
	axs_2[1].set_ylabel('Power (mW)')
	axs_2[1].set_title('TEC Power vs K_ev')
	axs_2[1].legend()
	axs_2[1].grid()
	axs_2[1].minorticks_on()
	plt.show()

	fig_3, axs_3 = plt.subplots(1, 2)
	axs_3[0].plot(-1 * dts_wrt_plate, pred_k_plate, label='K wrt plate', marker='o')
	axs_3[0].plot(-1 * dts_wrt_cu, pred_k_cu, label='K wrt Cu housing', marker='s')
	axs_3[0].scatter(10, 0.0233, color='purple', marker='P', label='K paper measurement')
	axs_3[0].set_title('K vs ∆T')
	axs_3[0].set_ylabel('K (W/K)')
	axs_3[0].set_xlabel('∆T (K)')
	axs_3[0].legend()
	axs_3[0].grid()
	axs_3[0].minorticks_on()

	axs_3[1].plot(pred_k_plate, power, label='K wrt plate', marker='o')
	axs_3[1].plot(pred_k_cu, power, label='K wrt Cu housing', marker='s')
	axs_3[1].scatter(0.0233, 420, color='purple', marker='P', label='K paper measurement')
	axs_3[1].set_xlabel('K (W/K)')
	axs_3[1].set_ylabel('Power (mW)')
	axs_3[1].set_title('TEC Power vs K')
	axs_3[1].legend()
	axs_3[1].grid()
	axs_3[1].minorticks_on()
	plt.show()


	k_fit_df = pd.DataFrame({
		'k_fit_wrt_cu': k_fit_cu,
		'k_fit_wrt_plate': k_fit_plate,
		'k_cu': pred_k_cu, 
		'k_plate': pred_k_plate,
		'k_ev_wrt_cu': k_ev_cu, 
		'k_ev_wrt_plate': k_ev_plate,
	}, index=power)

	return round(k_fit_df, 4)


def check_theory():	
	"""
	DEFUNCT
	Not entirely accurate
	Using paper values to test the theory.
	"""
# from the paper we know some information at -35 ºC:
	# 𝑅_𝑎𝑐 [Ω]= 2.58 ± 0.05
	# 𝛼 [mV/K] = 12.3 ± 0.2
	# 𝐾 [mW/K] = 23.3 ± 0.5
	# 𝐾𝑒𝑣 = 63 mW/◦C 
	# From the algorithm, we have a module temp of -35.4 ºC at a power of 114.167 mW, which is a total power of 2.74 W
	# From the elog at that power, V = 1.37 V and I = 2 A
	# From our calculations, T_c = Average plate temperature = -27.6 ºC
	# At that point, dT is -7.79
	alpha = 0.0123 * 24		# at -35.4 alpha = 0.012284; at -35 alpha = 0.0123; value that leads to closest approximation = 0.0125
	dT = -7.79
	R_ac = 2.58  * 24	# at -35.4 R_ac = 2.572; at -35 R_ac = 2.58; value that leads to closest approximation = 2.53
	I = 2
	print(f'Voltage fit: {voltage_fit(alpha, dT, R_ac, I)}')
	K = 0.0233
	K_ev = 0.063
	T_c = -27.6
	print(f'Qc fit: {qc_fit(alpha, I, T_c, R_ac, (K+K_ev), dT)}')


def check_voltage_fit():	
	"""
	DEFUNCT
	Checking accuracy of voltage fit.
	"""
# this dt was with respect to the micro-heaters actually
	# dts = np.array([6.313086, 8.720202, 11.336297, 13.96601, 16.042038, 17.665252, 19.240934]) 	# Th = plate
	dts = np.array([0, 2.407116, 5.023211, 7.652924, 9.728952, 11.352166, 12.927848])						# Th = module first temp
	cu_corrected_dts = 3.5 + dts
	actual_vs = np.array([0, 0.67, 1.37, 2.07, 2.76, 3.44, 4.14])
	# wrt copper housing
	copper_housing_pred_vs = np.array([0.315237044, 0.888036909, 1.451437111, 1.99570336, 2.502292169, 2.980732878, 3.446331208])
	# without copper housing correction
	pred_vs = np.array([0.140468373, 0.714616222, 1.279481438, 1.825220326, 2.33297171, 2.812321419, 3.278802131])
	# with copper housing
	# pred_vs = [0, 0.575231277, 1.141273981, 1.688196487, 2.196882278, 2.676962586, 3.144152502]
	# copper_housing_pred_vs = [0.174768672, 0.748651964, 1.313229655, 1.858679521, 2.366202737, 2.845374045, 3.311681579]
	plt.plot(-1 * dts, actual_vs, label='Actual voltage values')
	plt.plot(-1 * dts, pred_vs, label='Predicted voltage values')
	plt.plot(-1 * cu_corrected_dts, copper_housing_pred_vs, label='Predicted voltage corrected for copper housing')
	plt.legend()
	plt.xlabel('∆T (ºC)')
	plt.ylabel('Voltage (V)')
	plt.minorticks_on()
	plt.show()

	return copper_housing_pred_vs, pred_vs


def check_power_fit():	
	"""
	DEFUNCT
	Checking accuracy of power fit.
	"""
	# dts = np.array([6.313086, 8.720202, 11.336297, 13.96601, 16.042038, 17.665252, 19.240934]) 	# Th = plate
	dts = np.array([0, 2.407116, 5.023211, 7.652924, 9.728952, 11.352166, 12.927848])						# Th = module first temp
	# cu_corrected_dts = np.array([3.5, 5.907116, 8.523211, 11.152924, 13.228952, 14.852166, 16.427848])
	cu_corrected_dts = 3.5 + dts
	currents = np.array([0, 0.043478261, 0.086956522, 0.130434783, 0.173913043, 0.217391304, 0.260869565])
	actual_vs = np.array([0, 0.67, 1.37, 2.07, 2.76, 3.44, 4.14])
	copper_housing_pred_vs, pred_vs = check_voltage_fit()
	actual_powers = currents * actual_vs * 1000
	pred_powers = currents * pred_vs * 1000
	cu_corrected_pred_powers = currents * copper_housing_pred_vs * 1000
	
	plt.plot(-1 * dts, actual_powers, label='Actual power values')
	plt.plot(-1 * dts, pred_powers, label='Predicted power values')
	plt.plot(-1 * cu_corrected_dts, cu_corrected_pred_powers, label='Predicted power values (wrt copper housing)')
	plt.legend()
	plt.xlabel('∆T (ºC)')
	plt.ylabel('Power (mW)')
	plt.minorticks_on()
	plt.show()

	return actual_powers


def get_kfit_value():	
	"""
	DEFUNCT
	Checking accuracy of K_fit fit.
	"""
	# dts = np.array([6.313086, 8.720202, 11.336297, 13.96601, 16.042038, 17.665252, 19.240934]) 	# Th = plate
	dts = np.array([0, 2.407116, 5.023211, 7.652924, 9.728952, 11.352166, 12.927848])						# Th = module first temp
	# cu_corrected_dts = np.array([3.5, 5.907116, 8.523211, 11.152924, 13.228952, 14.852166, 16.427848])
	cu_corrected_dts = 3.5 + dts
	# nominal qc: 480mW per array
	qc = np.array([0.480])
	alphas = np.array([0.049933906, 0.049548768, 0.049130192, 0.048709438, 0.048377274, 0.04811756, 0.047865451])
	currents = np.array([0, 0.043478261, 0.086956522, 0.130434783, 0.173913043, 0.217391304, 0.260869565])
	T_cs = np.array([-30.413086, -32.820202, -35.436297, -38.06601, -40.142038, -41.765252, -43.340934])
	R_acs = np.array([10.67166341, 10.48711785, 10.28655056, 10.08493923, 9.925777087, 9.80133068, 9.680528393])
	K_fit = ((alphas * currents * T_cs) - (0.5 * R_acs * (currents**2)) - qc)/(dts)
	cu_corrected_k_fit = ((alphas * currents * T_cs) - (0.5 * R_acs * (currents**2)) - qc)/(cu_corrected_dts)

	print(f'kfit: {K_fit}')

	plt.plot(-1 * dts, K_fit, label='K_fit with ∆T')
	plt.plot(-1 * cu_corrected_dts, cu_corrected_k_fit, label='K_fit with ∆T (wrt Cu housing)')
	plt.ylabel('K_fit (K + K_ev) (W/K)')
	plt.xlabel('∆T (K)')
	plt.legend()
	plt.show()

	return K_fit, cu_corrected_k_fit


def kfit_vs_TEC_power():	
	"""
	DEFUNCT
	Checking accuracy of K_fit fit by plotting against TEC power and comparing with plot in the paper. 
	"""
	k_fit, cu_corrected_k_fit = get_kfit_value()
	actual_TEC_powers = check_power_fit()
	plt.plot(k_fit, actual_TEC_powers, label='K_fit with TEC power')
	plt.plot(cu_corrected_k_fit, actual_TEC_powers, label='K_fit with TEC power (corrected for Cu housing)')
	plt.title('K_Fit vs TEC Powers')
	plt.xlabel('K_fit (W/K)')
	plt.ylabel('TEC Power (mW)')
	plt.minorticks_on()
	plt.show()


if __name__ == '__main__':
# if you want to analyse the data from all the tests
	# for date_str in DATE_DICT.keys():

# else choose a particular date
		date_str = 'AUG_10'
		date = DATE_DICT[date_str]

		# # Format all the dataframes you need
		plate_df = reformat_df(date['TRAY_PLATE_FILENAME'], date['TRAY_PLATE_HEADER'])
		module_df = reformat_df(date['MODULE_FILENAME'], date['MODULE_HEADER'])
		cu_housing_df = reformat_df(date['CU_HOUSING_FILENAME'], date['CU_HOUSING_HEADER'])
		# lyso_df = reformat_df(date['LYSO_FILENAME'], date['LYSO_HEADER'])
		# t_inf_df = reformat_df(date['AMBIENT_FILENAME'], date['AMBIENT_HEADER'])
		# supply_df = reformat_df(date['CO2_SUPPLY_FILENAME'], date['CO2_SUPPLY_HEADER'])
		# plant_df = reformat_df(date['CO2_PLANT_FILENAME'], date['CO2_PLANT_HEADER'])
		

		# # Edit the DF list as relevant/necessary based on what you need - but make sure all of the DFs
		# you are making above start and end at the same time

		df_list = same_time_frame([plate_df, module_df, cu_housing_df])
		plate_df, module_df, cu_housing_df = df_list
		

		# # If you want to plot all the DFs you have
		# labels = [date['TRAY_PLATE_HEADER'], date['MODULE_HEADER'], date['CU_HOUSING_HEADER'], date['LYSO_HEADER']]
		# plot_everything(df_list, labels)


		# # Optional, not really required to do this step to get initial temperature
		# init_temp, init_temp_error, change_idx = correcting_init_temp(plate_df, date['TRAY_PLATE_HEADER'], module_df, date['MODULE_HEADER'], MODULE_PLANT_THRESHOLD, PLATE_PLANT_THRESHOLD)
		

		# # Do this entire block together
		step_mean_temps, errors, step_start_ends = find_step_avg_temp2(module_df, date['MODULE_HEADER'], FALL_THRESHOLD, RISE_THRESHOLD, 0)
		temps_powers, errors = compare_temp_power_dependencies_2(module_df, date['MODULE_HEADER'], date['ELOG_FILENAME'], 0)
		plot_temp_power_dependencies(temps_powers)
			# for a cooling cycle
		# dts_df = find_dt_power_cooling(cu_housing_df, date['CU_HOUSING_HEADER'], plate_df, date['TRAY_PLATE_HEADER'], module_df, date['MODULE_HEADER'], temps_powers, step_start_ends)
			# for an annealing cycle
		dts_df = find_dt_power_heating(cu_housing_df, date['CU_HOUSING_HEADER'], plate_df, date['TRAY_PLATE_HEADER'], module_df, date['MODULE_HEADER'], temps_powers, step_start_ends)
		print(dts_df)


		# # For theory exploration
		# rac_df_cu = predicted_rac_values(dts_df, date['ELOG_FILENAME'], 1)
		# actual_rac_cu = rac_df_cu['actual_rac'].to_numpy()
		# rac_df_plate = predicted_rac_values(dts_df, date['ELOG_FILENAME'], 0)
		# actual_rac_plate = rac_df_plate['actual_rac'].to_numpy()
		# # qc = -0.484 is specifically for Aug 2 test - update as necesssary
		# print(better_predicted_kfit(dts_df, date['ELOG_FILENAME'], lyso_df, date['LYSO_HEADER'], t_inf_df, date['AMBIENT_HEADER'], step_start_ends, qc=-0.484, rac_cu=actual_rac_cu, rac_plate=actual_rac_plate))

# -----------------------------------------------
# # NOW IRRELEVANT

# Scanning for window size
	# print(scanning_for_window_size(plate_df, TRAY_PLATE_HEADER, module_df, MODULE_HEADER, MODULE_PLANT_THRESHOLD))
	# print(scanning_for_window_size(plate_df, TRAY_PLATE_HEADER, module_df, MODULE_HEADER, PLATE_PLANT_THRESHOLD, MODULE_PLANT_THRESHOLD))
	# compare_rms_with_window_size(CO2_PLANT_FILENAME, CO2_PLANT_HEADER, TRAY_PLATE_FILENAME, TRAY_PLATE_HEADER)
	# compare_rms_with_window_size(TRAY_PLATE_FILENAME, TRAY_PLATE_HEADER, MODULE_FILENAME, MODULE_HEADER)


# Correcting temperatures
	# init_temp, init_temp_error, change_idx = correcting_init_temp(plant_df, CO2_PLANT_HEADER, plate_df, TRAY_PLATE_HEADER, PLATE_PLANT_THRESHOLD, MODULE_PLANT_THRESHOLD)
	# corr_devs, devs = correcting_temp()
	# init_temp, init_temp_error, change_idx = correcting_init_temp(plate_df, TRAY_PLATE_HEADER, module_df, MODULE_HEADER, MODULE_PLANT_THRESHOLD, MODULE_PLANT_THRESHOLD)

# Measuring cross-correlation
	# print(measure_cross_correlation(plant_df, CO2_PLANT_HEADER, plate_df, TRAY_PLATE_HEADER))
	# print(measure_cross_correlation(plate_df, TRAY_PLATE_HEADER, module_df, MODULE_HEADER))


# # PREVIOUS ITERATIONS OF CALCULATIONS: NOT USED, ALL DEFUNCT

# # FOR OSCILLATION:
	# fft_data = fft(resampled_temps.to_numpy())
	# freq = fftfreq(len(resampled_times.to_numpy()))
	# freq = freq[freq != 0]

	# print(f"Freq {freq}")
	# print(f"fft_data max: {max(np.abs(fft_data))}")
	# dominant_freq_index = np.argmax(np.abs(fft_data))
	# print(f"Dom freq idx: {dominant_freq_index}")
	# dominant_freq = freq[dominant_freq_index]
	# print(f"Dom freq: {dominant_freq}")
	# oscillation_period = 1 / np.abs(dominant_freq)

# # FAILED SCALING FACTOR CALCULATIONS:
	# max_noise_amp = (np.max(noise_fn) - np.min(noise_fn))/2
	# max_signal_amp = (np.max(signal_fn) - np.min(signal_fn))/2
	# scaling_factor = max_noise_amp/max_signal_amp

	# noise_amp, noise_freq = approximate_signal(noise_fn)
	# signal_amp, signal_freq = approximate_signal(signal_fn)
	# scaling_factor = noise_amp/signal_amp
