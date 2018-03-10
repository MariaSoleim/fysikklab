import iptrack
import math
import os
import numpy as np

g = 9.8214675 # +-0.0000004
m = 0.0027 #Fyll inn riktig masse
r = 0.02 #Fyll inn riktig radius
I = 2/3*m*r**2

_file = os.getcwd() + "/rapport/run.txt"

poly = iptrack.iptrack(_file)

data=np.loadtxt(_file)
x_values = data[:,1]


a_values = []
for x in x_values:
    alpha = iptrack.trvalues(poly, x)[3]
    a_values.append(g*math.sin(alpha)/(1+I/(m*r**2)))

for a in a_values:
    print(a)

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
