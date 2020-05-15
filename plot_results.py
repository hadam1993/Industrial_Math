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
	plotNormedRuntime(df)

def cleanColums(df):
	tmp = df.columns
	tmp = [bla.lstrip().replace(' ','').replace('.','') for bla in df.columns]
	df.columns = tmp

def cleanNetworkType(df):
	tmp = [('shallow' if 'Shallow' in bla else 'medium') for bla in df['NetworkType']]
	df['NetworkType'] = tmp

def plotNormedRuntime(df):
	idx = 'CleaningIndex'
	runtime = 'Runtime'
	epochs = 'MinEpoch'
	shallow = df.loc[df['NetworkType']=='shallow']
	medium = df.loc[df['NetworkType']=='medium']
	mpp.plot(shallow[idx],shallow[runtime]/shallow[epochs],c=shallowCol,marker='x',label='Shallow Network')
	mpp.plot(medium[idx],medium[runtime]/medium[epochs],c=mediumCol,marker='.',label='Medium Network')
	mpp.title('Run time per epoch over data cleaning')
	mpp.ylabel('Run time per epoch [s]')
	#mpp.ylim()
	mpp.savefig('images/plot_timePerEpoch.png')

def plotColumns(df, idx='CleaningIndex', cols=[ 'TestAccuracy', 'MinEpoch', 'MinValidationLoss', 'Runtime']):
	shallow = df.loc[df['NetworkType']=='shallow']
	medium = df.loc[df['NetworkType']=='medium']
	for col in cols:
		mpp.plot(shallow[idx],shallow[col],c=shallowCol,marker='x',label='Shallow Network')
		mpp.plot(medium[idx],medium[col],c=mediumCol,marker='.',label='Medium Network')
		mpp.xlabel('Cleaning level')
		mpp.legend()
		if col == 'Runtime':
			mpp.title('Dependence of run time on data cleaning')
			mpp.ylabel('Run time [s]')
			#mpp.ylim()
		elif col == 'TestAccuracy':
			mpp.title('Dependence of test accuracy on data cleaning')
			mpp.ylabel('Test accuracy')
			mpp.ylim(0.5,1)
		elif col == 'MinEpoch':
			mpp.title('Dependence of optimal training epoch on data cleaning')
			mpp.ylabel('Optimal training epoch')
			mpp.ylim(0,100)
		elif col == 'MinValidationLoss':
			mpp.title('Dependence of minimal validation loss on data cleaning')
			mpp.ylabel('Validation loss at optimal epoch')
			mpp.ylim(.18,.68)
		mpp.savefig('images/plot_{}.png'.format(col))
		mpp.close()

if __name__ == '__main__':
	main()