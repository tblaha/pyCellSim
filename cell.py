import numpy as np

class Cell():
	# https://www.researchgate.net/figure/Battery-Equivalent-Circuit-Cell-Model_fig7_289205761
	def SampleChargeCurve_continuous(Q_Ah):
		A = 3.8 # V
		B = 0.3 # V
		C = -0.5 # 1/Ah
		D = -4.8e-2 # V
		E = 2.0 # 1/Ah
		F = 6 # Ah
		G = -15e-3 # V/Ah

		VOC = lambda Q_Ah, A,B,C,D,E,F,G: \
			A + B*np.exp(C*Q_Ah) + D*np.exp(E*(Q_Ah-F)) + G*Q_Ah

		return VOC(Q_Ah, A,B,C,D,E,F,G)

	def SampleChargeCurve(SoC):
		SoCs = np.linspace(0.0, 1.0, 20)
		VOCs = np.array([3.17108465, 3.45962173, 3.59913086, 3.66906171, 3.70670361, 3.7295988 , 3.74604896, 3.7600419 , 3.77356401, 3.78768823, 3.80308904, 3.82028999, 3.83978717, 3.86211607, 3.88789393, 3.91785265, 3.95287023, 3.99400473, 4.04253351, 4.09999971])
		N = 20

		eps = 1e-6
		if np.isscalar(SoC):
			idx = int( np.floor( np.clip(SoC, SoCs[0], SoCs[-2]+eps)*(N-1) ) )
			VOC = VOCs[idx] + ( VOCs[idx+1] - VOCs[idx] ) / ( SoCs[idx+1] - SoCs[idx] ) * (SoC - SoCs[idx])
		else:
			VOC = np.zeros_like(SoC)
			for i in range(len(SoC)):
				idx = int( np.floor( np.clip(SoC[i], SoCs[0], SoCs[-2]+eps)*(N-1) ) )
				VOC[i] = VOCs[idx] + ( VOCs[idx+1] - VOCs[idx] ) / ( SoCs[idx+1] - SoCs[idx] ) * (SoC[i] - SoCs[idx])

		return VOC

	def __init__(self, Rtotal, RST, RLT, tauST, tauLT, Ah_cap):
		self.R0 = Rtotal - RST - RLT
		self.RST = RST
		self.RLT = RLT
		self.tauST = tauST
		self.tauLT = tauLT
		self.CST = tauST / RST
		self.CLT = tauLT / RLT
		self.Ah_cap = Ah_cap

		self.SoC = 1.0
		self.VOC = Cell.SampleChargeCurve(self.SoC)
		self.V0 = self.VST = self.VLT = 0.0
		self.IL = 0.0
		self.Q = 0

		self.solver = FwdEuler(self._getStateDerivative)

	def getSteadyStateVoltage(self):
		return self.VOC
	
	def getLeadVoltage(self):
		return self.VOC - self.IL*self.R0 - self.VST - self.VLT

	def getSoC(self):
		return 1.0 - self.Q/3600 / self.Ah_cap

	def getQ(self):
		return self.Q

	def setCurrent(self, I):
		self.IL = I

	def setSoC(self, SoC):
		self.SoC = SoC
		self.Q = (1.0 - self.SoC) * self.Ah_cap * 3600
		self.VOC = Cell.SampleChargeCurve(SoC)
		self.V0 = self.VST = self.VLT = 0.0

	def _getState(self):
		return np.array([self.Q, self.VST, self.VLT])

	def _getStateDerivative(self):
		dQ_dt = self.IL
		dVST_dt = ( self.IL - self.VST/self.RST ) / self.CST
		dVLT_dt = ( self.IL - self.VLT/self.RLT ) / self.CLT
		
		return np.array([dQ_dt, dVST_dt, dVLT_dt])

	def step(self, DT):
		Dy = self.solver.getDy(DT)

		self.Q += Dy[0]

		self.VOC = Cell.SampleChargeCurve(1.0-self.Q/3600/self.Ah_cap)
		self.V0 = self.IL/self.R0
		self.VST += Dy[1]
		self.VLT += Dy[2]


class FwdEuler():
	def __init__(self, fun):
		self.fun = fun
	
	def getDy(self, DT):
		return DT * self.fun()
