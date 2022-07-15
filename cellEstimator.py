import numpy as np
from scipy.optimize import fsolve
from cell import Cell

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

	def _updateSoC(self):
		fun = lambda x: Cell.SampleChargeCurve(x) - self.VOC
		self.SoC = 1.0 - fsolve(fun, (1.0-self.SoC)*self.Ah_cap)/self.Ah_cap
	
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
		Qk = np.diag([1e0, 1e0])
		Rk = 1e5
		uk = self.IL
		zk = self.VL

		xk1k = F@xkk + B*uk
		Pk1k = F@Pkk@F.T + Qk
		yk = zk - H@xk1k
		Sk = H@Pk1k@H.T + Rk
		Kk = Pk1k@H/Sk

		xkk = xk1k + Kk*yk
		self.VOC = xkk[1]
		self.P = (np.eye(2) - Kk@H)@Pk1k

		self._updateSoC()

