import matplotlib.pyplot as plt
import numpy as np
import os
import math
from openpyxl import Workbook, load_workbook


g = 9.8214675  # +-0.0000004
m = 0.0027  # Fyll inn riktig masse
r = 0.02  # Fyll inn riktig radius
h = 0.01
inertia = 2/3*m*r**2


def euler(v, alpha):
    return v + ((0.03/5) * (g*math.sin(alpha)))


def trvalues(p, x):
    y = np.polyval(p, x)
    dp = np.polyder(p)
    dydx = np.polyval(dp, x)
    ddp = np.polyder(dp)
    d2ydx2 = np.polyval(ddp, x)
    alpha = np.arctan(-dydx)
    R = (1.0+dydx**2)**1.5/d2ydx2
    return [y, dydx, d2ydx2, alpha, R]


def get_alpha(poly, point):
    return trvalues(poly, point)[3]


class Sheet:
    def __init__(self, sheet):
        self.sheet = sheet
        self.t = self.col('A')  # time
        self.x = self.col('B')  # x-position
        self.y = self.col('C')  # y-position

        # set up the sheets polynomial fit
        self.poly = self.iptrack()

        # for use by eulers method, initialize default values
        self.velocity = [0]
        self.distance = [self.x[0]]
        self.acceleration = [self.compute_acceleration()]

        # initialize the first acceleration value

    def plot_x(self): plt.plot(self.t, self.x)

    def plot_y(self): plt.plot(self.t, self.y)

    def plot_euler(self):
        # actual values
        plt.plot(self.t, self.x)
        # numerically computed values
        plt.plot(self.t, self.distance)

    def col(self, col_name):
        return [float(c.value) for c in self.sheet[col_name]
                if c.value is not None]

    def get_t(self): return self.t

    def get_x(self): return self.x

    def get_y(self): return self.y

    def iptrack(self):
        return np.polyfit(self.x, self.y, 15)

    def compute_acceleration(self, point=None):
        if point is None:
            # choose the first point as default initial value
            point = self.x[0]
        alpha = trvalues(self.poly, point)[3]
        return g*math.sin(alpha)/(1+(inertia/m*r**2))

    def euler_iteration(self):
        # fetch the first items in the experimental data sets
        x = self.distance[-1]
        v = self.velocity[-1]
        alpha = get_alpha(self.poly, x)

        x_next = x + h*v
        v_next = v + h*g*math.sin(alpha)/(1 + (inertia/m*r**2))
        a_next = self.compute_acceleration(x_next)

        self.distance.append(x_next)
        self.velocity.append(v_next)
        self.acceleration.append(a_next)

    def compute_euler(self):
        for i in range(len(self.t)-1):
            self.euler_iteration()


wb = Workbook()
data_src = os.path.join(os.getcwd(), "lab2_data")
linear_book, sine_book, steep_book = [], [], []
data_dict = {"Linear": [], "Sine": [], "Steep": []}


def init():
    for curve_type in os.listdir(data_src):
        _path = os.path.join(data_src, curve_type)
        if "xlsx" not in _path:
            continue
        book = load_workbook(_path)
        for sheet in book.sheetnames:
            data_dict[curve_type[:-5]].append(Sheet(book[sheet]))


init()


def plot_sheet(s):
    print(s)
    # iterate over every sheet/tab in the xls
    poly = s.iptrack()
    x_vals = [x for x in s.get_x() if x is not None]
    print(x_vals)
    #  accel = [trvalues(poly, x) for x in x_vals]
    accel = []
    for x in x_vals:
        if x is not None:
            alpha = trvalues(poly, x)
            accel.append(g*math.sin(alpha)/(1+inertia/(m*r**2)))
    s.set_acc(accel)
    plt.subplot(5, 2, fig_num+1)
    s.plot_acc()
    s.plot_y()


for k, v in data_dict.items():
    # iterate over every xls document
    print(k)
    fig_num = 0
    for s in v:
        plt.subplot(5, 2, fig_num+1)
        s.compute_euler()
        s.plot_euler()
        fig_num += 1
    plt.show()
    # do something with s here
    #  plt.title(k)
    #  plt.ylabel('acceleration m/s^2')
    #  plt.xlabel('time')
    plt.show()
