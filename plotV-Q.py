from cell import Cell
import numpy as np
from matplotlib import pyplot as plt

#%% Plot voltage curve

Q_Ah = np.linspace(0, 7.2, 1000)

f = plt.figure()
ax = f.subplots()
ax.plot(Q_Ah, Cell.SampleChargeCurve(Q_Ah))
ax.grid()
f.show()