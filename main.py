import numpy as np
from cell import Cell
from cellEstimator import CellEstimatorSimple, CellEstimatorKalman
import matplotlib.pyplot as plt
import pandas as pd



dat = pd.read_csv("data/FSN22_EnduFixed.csv")

# e = CellEstimatorSimple(
# 	Rtotal=4e-3 - 0.5e-3,
# 	VL=4.0,
# 	Ah_cap=7.200,
# 	)
e = CellEstimatorKalman(
	Rtotal=4.5e-3, # WARNING!!! TODO!! This is "calibrated" for the broken cell 35. Actual number probably lower
	VL=3.0, # TODO! This initial value seems a bit important, since the Kalman acts slow. Or maybe increase initial Pkk?
	Ah_cap=7.200, # only needed to compute SoC fraction. Not important to Kalman internals 
	)


#%% run Kalman over FSN22 data

DT = 0.02 # sec. Can be 0.004, but that would be slow to compute
t = 27. # starting time
T_end = 1650.

ts = [t]
VL = [dat["CellVoltages.MinVoltage"].loc[0]] # lead voltage
IL = [0.] # lead current
SoC_est = [e.getSoC()]
VOC_est = [VL[0]] # internal "steady state" voltage

I_apply = 0.
VL_meas = 0.
idx_I = 0
idx_V = 0
while t < T_end:

	# no, thanks: correct resampling of Marple data
	# yes please: ugly first order holds
	while dat["time"].loc[idx_I] < t-0.3 and idx_I < dat.index[-1]: # 300ms delay seems best
		idx_I += 1
		I_apply = I_apply if np.isnan(dat["ControlsOut.TS_current"].loc[idx_I])\
			else 0.5*dat["ControlsOut.TS_current"].loc[idx_I] # 0.5 because 2P
	while dat["time"].loc[idx_V] < t and idx_V < dat.index[-1]:
		idx_V += 1
		VL_meas = VL_meas if np.isnan(dat["CellVoltages.MinVoltage"].loc[idx_V])\
			else dat["CellVoltages.MinVoltage"].loc[idx_V]

	IL.append(I_apply)
	VL.append(VL_meas)

	# estimator
	e.setCurrent(I_apply)
	e.setLeadVoltage(VL_meas)
	e.step(DT)

	VOC_est.append(e.getSteadyStateVoltage())
	SoC_est.append(e.getSoC())

	t += DT
	ts.append(t)


#%% Plotting

f = plt.figure()
axs = f.subplots(4, 1)
axs[0].plot(ts, IL); axs[0].grid(); axs[0].set_ylabel("Lead Current [A]")
axs[1].plot(ts, VL); axs[1].grid(); axs[1].set_ylabel("Lead Voltage [V]")
axs[2].plot(ts, VOC_est); axs[2].grid(); axs[2].set_ylabel("Internal Voltage est [V]")
axs[3].plot(ts, SoC_est); axs[3].grid(); axs[3].set_ylabel("SoC est [-]")
axs[3].set_xlabel("Time [s]")
f.show()
