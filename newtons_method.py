import iptrack
import math
import os
import numpy as np
import matplotlib.pyplot as plt

g = 9.8214675 # +-0.0000004
m = 0.0027 #Fyll inn riktig masse
r = 0.02 #Fyll inn riktig radius
I = 2/3*m*r**2
h = 0.01

_file = os.getcwd() + "/rapport/run.txt"

poly = iptrack.iptrack(_file)

data=np.loadtxt(_file)

# Returns the acceleration using an x-value.
def getA(x):
    alpha = iptrack.trvalues(poly, x)[3]
    a = g*math.sin(alpha)/(1+(I/m*r**2))
    return a

# Adding initializing values to lists
t_values = np.arange(0, 1.27, 0.01)
x_values = [data[:,1][0]]
v_values = [0]
a_values = [getA(x_values[-1])]

# Eulers method.
for i in range(126):
    x = x_values[-1]
    v = v_values[-1]

    alpha = iptrack.trvalues(poly, x)[3]

    x_next = x + h*v
    v_next = v + h*g*math.sin(alpha)/(1+(I/m*r**2))
    a_next = getA(x_next)

    v_values.append(v_next)
    x_values.append(x_next)
    a_values.append(a_next)


def print_table():
    print("t\tx\tv\ta")
    for i in range(127):
        print(str(t_values[i]) + "\t" + str(round(x_values[i],3)) + "\t" + str(round(v_values[i],3)) + "\t" + str(round(a_values[i],3)))

print_table()
experimental_t_values = np.loadtxt(_file)[:,0]
experimental_x_values = np.loadtxt(_file)[:,1]
plt.plot(t_values, x_values)
plt.plot(experimental_t_values, experimental_x_values)
plt.show()
