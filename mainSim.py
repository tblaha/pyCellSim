import numpy as np
from cell import Cell
from cellEstimator import CellEstimatorSimple, CellEstimatorKalman
import matplotlib.pyplot as plt
#import pandas as pd


c = Cell(
	Rtotal=4e-3,
	RST=0.3e-3,
	RLT=0.1e-3,
	tauST=0.5,
	tauLT=30.0,
	Ah_cap=7.200,
	)
c.setSoC(1.0)

# e = CellEstimatorSimple(
# 	Rtotal=4e-3 - 0.5e-3,
# 	VL=3.9,
# 	Ah_cap=7.200,
# 	)
e = CellEstimatorKalman(
	Rtotal=4e-3 - 0.3e-3,
	VL=3.8,
	Ah_cap=7.200,
	)


#%% apply step input

c.setSoC(0.7)
DT = 0.004 # sec. sensible maximum: 0.2*min(tauST, tauLT)
T_step = 1. # sec
T_stop = 20.
T_end = 40.
t = 0.
t_last_V = t

I_step = 50. # Amp

ts = [t]
VOC_GT = [c.getSteadyStateVoltage()]
VL = [c.getLeadVoltage()]
VL_GT = [c.getLeadVoltage()]
VL_est = [e.getLeadVoltage()]
SoC_GT = [c.getSoC()]
SoC_est = [c.getSoC()]
It = [c.getQ()]
IL = [0.]
VOC_est = [c.getLeadVoltage()]


I_apply = 0.
VL_meas = c.getLeadVoltage()
VL_meas_delay = c.getLeadVoltage()
idx = 0
while t < T_end:
	if t > T_step and t < T_stop:
		I_apply = I_step
	else:
		I_apply = 0.
	IL.append(I_apply)
	# cell
	c.setCurrent(I_apply)
	c.step(DT)
	
	if t-t_last_V >= 0.2:
		VL_meas = VL_meas_delay
		VL_meas_delay = c.getLeadVoltage()
		t_last_V = t
		e.setLeadVoltage(VL_meas)

	VL.append(VL_meas)
	VL_GT.append(c.getLeadVoltage())

	VOC_GT.append(c.getSteadyStateVoltage())
	SoC_GT.append(c.getSoC())
	It.append(c.getQ())

	# estimator
	e.setCurrent(I_apply)
	e.step(DT)
	VL_est.append(e.getLeadVoltage())

	VOC_est.append(e.getSteadyStateVoltage())
	SoC_est.append(e.getSoC())

	t += DT
	ts.append(t)



f = plt.figure()
axs = f.subplots(4, 1)
axs[0].plot(ts, IL); axs[0].grid(); axs[0].set_ylabel("Lead Current [A]")
axs[0].legend(["Lead Current Measurement"])

axs[1].plot(ts, VL); axs[1].grid(); axs[1].set_ylabel("Lead Voltage [V]")
axs[1].plot(ts, VL_GT)
axs[1].plot(ts, VL_est)
axs[1].legend(["Lead Voltage Measurement", "Lead Voltage GT", "Lead Voltage est"])

axs[2].plot(ts, VOC_GT); axs[2].grid(); axs[2].set_ylabel("VOC [V]")
axs[2].plot(ts, VOC_est)
axs[2].legend(["Internal Voltage Ground Truth", "Internal Voltage Estimated"])

axs[3].plot(ts, SoC_GT); axs[3].grid(); axs[3].set_ylabel("SoC [-]")
axs[3].plot(ts, SoC_est)
axs[3].legend(["SoC Ground Truth", "SoC Estimated"])
axs[3].set_xlabel("Time [s]")

f.show()

