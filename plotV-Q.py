from cell import Cell
import numpy as np
from matplotlib import pyplot as plt

#%% Plot voltage curve

SoC = np.linspace(0, 1.0, 1000)

f = plt.figure()
ax = f.subplots()
ax.plot(SoC, Cell.SampleChargeCurve(SoC))
ax.grid()
f.show()

