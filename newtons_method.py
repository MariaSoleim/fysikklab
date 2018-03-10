import iptrack
import math
import os
import numpy as np

g = 9.81 #Fyll inn riktig g
m = 0.01 #Fyll inn riktig masse
r = 0.02 #Fyll inn riktig radius
I = 2/3*m*r**2

_file = os.getcwd() + "/rapport/run.txt"
data=np.loadtxt(_file)
x_values = data[:,1]
poly = iptrack.iptrack(_file)

for x in x_values:
    a_values = []
    y, dydx, d2ydx2, alpha, R = iptrack.trvalues(poly, x)
    a_values.append(g*math.sin(alpha)/(1+I/(m*r**2)))

print(a_values)

"""
x = 0.01229306768 #Hente ut denne verdien fra skjema med kode
_file = os.getcwd() + "/rapport/run.txt"
poly = iptrack(_file)
y, dydx, d2ydx2, alpha, R = iptrack.trvalues(poly, x)

s = 0


s_next = s + v*dt
v = v + g*sin(alpha)/(1+I/(m*r**2))

x = x + (s_next - s)*math.cos(alpha)
y = y - (s_next - s)*math.sin(alpha)
"""
