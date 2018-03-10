import matplotlib.pyplot as plt
import numpy as np
import os
from openpyxl import Workbook, load_workbook

def iptrack(x, y):
    return np.polyfit(x, y, 15)


def trvalues(p, x):
    y = np.polyval(p, x)
    dp = np.polyder(p)
    dydx = np.polyval(dp, x)
    ddp = np.polyder(dp)
    d2ydx2 = np.polyval(ddp, x)
    alpha = np.arctan(-dydx)
    R = (1.0+dydx**2)**1.5/d2ydx2
    return [y, dydx, d2ydx2, alpha, R]

class Sheet:
    def __init__(self, sheet):
        self.sheet = sheet
        self.t = self.col('A')  # time
        self.x = self.col('B')  # x-position
        self.y = self.col('C')  # y-position

    def plot(self):
        plt.plot(self.t, self.y)

    def col(self, col_name):
        return [c.value for c in self.sheet[col_name]]

    def get_t(self):
        return self.t

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


wb = Workbook()
data_src = os.path.join(os.getcwd(), "lab2_data")
linear_book, sine_book, steep_book = [], [], []
data_dict = {"Linear": [], "Sine": [], "Steep": []}

for curve_type in os.listdir(data_src):
    _path = os.path.join(data_src, curve_type)
    if "xlsx" not in _path:
        continue
    book = load_workbook(_path)
    for sheet in book.sheetnames:
        data_dict[curve_type[:-5]].append(Sheet(book[sheet]))

for k, v in data_dict.items():
    print(k)
    fig_num = 0
    for i in v:
        plt.subplot(5, 2, fig_num+1)
        i.plot()
        fig_num += 1
        print(i)
    plt.show()
