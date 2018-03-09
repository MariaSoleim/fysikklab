import matplotlib.pyplot as plt
import numpy as np
import openpyxl

from openpyxl import Workbook
wb = Workbook()
book = openpyxl.load_workbook('/rapport/lab2_linear0.xlsx')
ws = wb.active
sheet = book.active

t = sheet['A']
x = sheet['B']
y = sheet['C']

new_t = []
for item in t:
    new_t.append(item.value)

new_x = []
for item in x:
    new_x.append(item.value)

new_y = []
for item in y:
    new_y.append(item.value)

plt.plot(new_x, new_y)
plt.show()
