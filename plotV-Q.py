from cell import Cell
import numpy as np
from matplotlib import pyplot as plt

#%% Plot voltage curve

SoC = np.linspace(0, 1.0, 1000)*7.200

f = plt.figure()
ax = f.subplots()
ax.plot(np.flipud(SoC/7.2), Cell.SampleChargeCurve_continuous(SoC))
ax.grid()
f.show()

