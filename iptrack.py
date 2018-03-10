# iptrack - interpolate track
#
# SYNTAX
# p=iptrack(filename)
#
# INPUT
# filename: data file containing exported tracking data on the standard
# Tracker export format
#
# mass_A
# t	x	y
# 0.0	-1.0686477620876644	42.80071293284619
# 0.04	-0.714777136706708	42.62727536827738
# ...
#
# OUTPUT
# p=iptrack(filename) returns the coefficients of a polynomial of degree 15
# that is the least square fit to the data y(x). Coefficients are given in
# descending powers.

import numpy as np
import matplotlib as plot
import os

_file = os.getcwd() + "/rapport/run.txt"

def iptrack(filename):
	data=np.loadtxt(filename,skiprows=2)
	print([round(x,10) for x in data[:,2]])
	return np.polyfit(data[:,1],data[:,2],15)

def trvalues(p,x):
	y=np.polyval(p,x)
	dp=np.polyder(p)
	dydx=np.polyval(dp,x)
	ddp=np.polyder(dp)
	d2ydx2=np.polyval(ddp,x)
	alpha=np.arctan(-dydx)
	R=(1.0+dydx**2)**1.5/d2ydx2
	return [y,dydx,d2ydx2,alpha,R]

if __name__ == "__main__":
	poly = iptrack(_file)
	#print(poly)
	#print(trvalues(poly, 0.01229306768))
	#print(trvalues(poly, 0.5))
