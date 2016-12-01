from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

'''LOAD & ClEAN data'''

def add_datum(rel_time, datum):
	'''calculate vector of absolute time'''
	temp = '00:' + rel_time
	temp = pd.to_timedelta(temp)
	abs_time = temp + datum
	return abs_time

def load_data(fname):
	'''return a datetime-indexed dataframe'''
	df = pd.read_csv(fname)
	date_str, time_str = filter(lambda x: x.isdigit(), fname.split('_'))
	datum = datetime.strptime(date_str+time_str, '%Y%m%d%H%M%S')
	dt_index = [add_datum(rel_time, datum) for rel_time in df.Timestamp]
	df.index = dt_index
	return df.drop('Timestamp', axis = 1)

'''IDENTIFY key times in data: touchdown, end braking, apex'''

def find_touchdown(df, feature, zval):
	diffed = df[feature]#.diff()
	i = 0
	for time, val in diffed.iteritems():
		if i < 12:
			i += 1
			continue
		window = diffed.loc[:time]
		window = window[:-1]
		std = window.std()
		mean = window.mean()
		if (val - mean)/ std >= zval:	
			return time
		i += 1		
	return None

def find_apex(decel):
	res = []
	for t in decel.index[10::10]:
		left = decel[:t]['accelY']
		right = decel[t:]['accelY']
		left_mod = OLS(left, add_constant(range(len(left)))).fit()
		right_mod = OLS(right, add_constant(range(len(right)))).fit()
		ssrs = [t,left_mod.ssr, right_mod.ssr]
		res.append(ssrs)
	apex = min(res, key = lambda x: x[1] + x[2])[0]
	return apex	

def segment_landing(df):
	touchdown = find_touchdown(df, 'accelY', 3)
	end_braking = find_touchdown(df[::-1], 'accelY', 3)
	decel = df[touchdown:end_braking]
	apex = find_apex(decel)
	return touchdown, end_braking, apex

''' Label the segments'''

def label_segments(df, touchdown, end_braking, apex):
	labels = []
	for t in df.index:
		if t < touchdown:
			label = 'Flying'
		elif t <= apex:
			label = 'Touchdown'
		elif t <= end_braking:
			label = 'Braking'
		else:
			label = 'Rolling'
		labels.append(label)
	df['label'] = labels
	return df
	
'''Visualize data'''

def plot_landing(df, feature, fname):
	#for feature in ['accelY']:#, 'accelZ']:
	for label in df.label.unique():
		df[df['label'] ==label][feature].plot(label = label)
	plt.legend()
	plt.ylabel(feature)
	plt.title('%s for %s' % (feature, fname))
	plt.show()

''' Work the magic '''

def label_data(fname, feature = 'accelY', return_df = False):
	df = load_data(fname)
	touchdown, end_braking, apex = segment_landing(df)
	df = label_segments(df, touchdown, end_braking, apex)
	plot_landing(df, feature, fname)
	if return_df:
		return df

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("fname", help="give the name of a file")
	args = parser.parse_args()
	label_data(args.fname)
	
# fname = 'BDL_DEN_landing_mlt_20161126_092237_56s.csv'
# fname = 'DEN_SFO_landing_mlt_20161126_153549_76s.csv'
# fname = 'SFO_DEN_landing_mlt_20161117_220658_76s.csv'
