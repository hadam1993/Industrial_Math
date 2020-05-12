#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as mpp

shallowCol = 'r'
mediumCol = 'b'

def main():
	df = pd.read_csv('kaggle_results.csv')
	cleanColums(df)
	cleanNetworkType(df)
	plotColumns(df)

def cleanColums(df):
	tmp = df.columns
	tmp = [bla.lstrip().replace(' ','').replace('.','') for bla in df.columns]
	df.columns = tmp

def cleanNetworkType(df):
	tmp = [('shallow' if 'Shallow' in bla else 'medium') for bla in df['NetworkType']]
	df['NetworkType'] = tmp

def plotColumns(df, idx='CleaningIndex', cols=[ 'TestAccuracy', 'MinEpoch', 'MinValidationLoss', 'Runtime']):
	shallow = df.loc[df['NetworkType']=='shallow']
	medium = df.loc[df['NetworkType']=='medium']
	for col in cols:
		mpp.plot(shallow[idx],shallow[col],c=shallowCol,marker='x',label='Shallow Network')
		mpp.plot(medium[idx],medium[col],c=mediumCol,marker='.',label='Medium Network')
		mpp.title('Dependence of {} on data cleaning'.format(col))
		mpp.xlabel('Cleaning level')
		mpp.legend()
		if col == 'RunTime':
			mpp.ylabel('{} [s]'.format(col))
			#mpp.ylim()
		elif col == 'TestAccuracy':
			mpp.ylabel('{} [%]'.format(col))
			mpp.ylim(0.5,1)
		elif col == 'MinEpoch':
			mpp.ylabel(col)
			mpp.ylim(0,100)
		elif col == 'MinValidationLoss':
			mpp.ylabel('{} [%]'.format(col))
		mpp.savefig('plot_{}.png'.format(col))
		mpp.close()

if __name__ == '__main__':
	main()