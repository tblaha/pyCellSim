import numpy as np
#from scipy.optimize import fsolve
from cell import Cell
from copy import deepcopy

class CellEstimatorSimple():
	def __init__(self, Rtotal, VL, Ah_cap):
		self.R0 = Rtotal
		self.IL = 0.
		self.Ah_cap = Ah_cap
		self.SoC = 1.0
		self.VOC = self.VL = VL
	
	def setCurrent(self, I):
		self.IL = I
	
	def setSoC(self, SoC):
		self.SoC = SoC
	
	def setLeadVoltage(self, VL):
		self.VL = VL
	
	def getSteadyStateVoltage(self):
		return self.VOC

	def getSoC(self):
		return self.SoC

	def step(self, DT):
		self.VOC = self.VL + self.IL*self.R0

		fun = lambda x: Cell.SampleChargeCurve(x) - self.VOC
		self.SoC = 1.0 - fsolve(fun, (1.0-self.SoC)*self.Ah_cap)/self.Ah_cap


class CellEstimatorKalman():
	def __init__(self, Rtotal, VL, Ah_cap):
		self.R0 = Rtotal
		self.IL = 0.
		self.Ah_cap = Ah_cap
		self.VOC = self.VL = VL
		self.SoC = 1.0 # needed as initial guess to _updateSoC()
		self._updateSoC()
		self.P = 1*np.eye(2)
		self.new_voltage = True
		self.t_since_last_VL = 0
		self.I_list = np.zeros((50, 1))
		self.VL_est = VL
		self.ILk = 0

	def _updateSoC(self):
		SoCs = np.linspace(0, 7.2, 20) / self.Ah_cap
		VOCs = np.array([3.17108465, 3.45962173, 3.59913086, 3.66906171, 3.70670361, 3.7295988 , 3.74604896, 3.7600419 , 3.77356401, 3.78768823, 3.80308904, 3.82028999, 3.83978717, 3.86211607, 3.88789393, 3.91785265, 3.95287023, 3.99400473, 4.04253351, 4.09999971])
		N = 20

		eps = 1e-6
		idx = int( np.floor( np.clip(self.SoC, SoCs[0], SoCs[-2]+eps)*(N-1) ) )
		found = False
		window = 1
		if VOCs[-1] <= self.VOC:
			found = True
			SoC = SoCs[-1]
		if VOCs[0] >= self.VOC:
			found = True
			SoC = SoCs[0]

		while not found:
			if VOCs[idx+window] > self.VOC:
				if VOCs[idx] <= self.VOC:
					found = True
					SoC = SoCs[idx] + (SoCs[idx+window]-SoCs[idx]) / (VOCs[idx+window] - VOCs[idx]) * (self.VOC - VOCs[idx])
				else:
					idx = idx - 1
			else:
				idx = idx + 1

		self.SoC = SoC


	def setCurrent(self, I):
		self.IL = I
	
	def setSoC(self, SoC):
		self.SoC = SoC
	
	def setLeadVoltage(self, VL):
		self.VL = VL
		self.new_voltage = True
	
	def getSteadyStateVoltage(self):
		return self.VOC

	def getLeadVoltage(self):
		return self.VL_est

	def getSoC(self):
		return self.SoC

	def step(self, DT):
		# https://en.wikipedia.org/wiki/Kalman_filter#Details
		# 
		# System Definition
		# u = IL  # current at leads
		# x = [VL; VOC]
		# x+ = [0 1; 0 1] x + [-R0; 0] u  # could include feedforward based on V-Q slope in B matrix instead of 0
		# y = [1 0] x

		F = np.array([[0, 1], [0, 1]])
		B = np.array([-self.R0, 0]) # potentially add V-Q curve slope?
		H = np.array([1, 0])

		xkk = np.array([self.VL, self.VOC])
		Pkk = self.P
		Qk = np.diag([2e1, 2e1])
		Rk = 1e4
		zk = self.VL

		self.t_since_last_VL += DT
		self.I_list[1:] = self.I_list[:-1]
		self.I_list[0] = self.IL
		uk = self.I_list[-1]

		if self.new_voltage:
			self.new_voltage = False
			Rk *= self.t_since_last_VL**2
			Qk *= self.t_since_last_VL**2
			B = np.array([-self.R0, -15e-3*self.t_since_last_VL/3600]) # potentially add V-Q curve slope?
			self.t_since_last_VL = 0

			xk1k = F@xkk + B*uk
			Pk1k = F@Pkk@F.T + Qk
			yk = zk - H@xk1k
			Sk = H@Pk1k@H.T + Rk
			Kk = Pk1k@H/Sk
			xkk = xk1k + Kk*yk
			self.P = (np.eye(2) - Kk@H)@Pk1k

			self.VOC = xkk[1]
			self._updateSoC()

			self.ILk = deepcopy(self.I_list[-1])
			self.VL_est = self.VL - (self.IL - self.ILk)*self.R0
		
		else:
			VL_est_ref = self.VL - (self.IL - self.ILk)*self.R0
			alpha = DT/(DT+0.02)
			self.VL_est = (1-alpha) * self.VL_est + alpha * VL_est_ref


