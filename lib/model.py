import numpy as np

def sigmoid(z,non_linearity):
	if non_linearity == 'logit':
		return 1/(1+np.exp(-z))
	if non_linearity == 'tanh':
		return np.tanh(z)


def initialise_parameters(v,h):
	weights = np.random.randn(v,h)/np.sqrt(v*h)
	bias = 0.01 * np.random.randn(h)
	return weights,bias


def softmax(var):
	exponent = np.exp(var)
	return exponent/sum(exponent)

def deriv_sigmoid(val,non_linearity):
	if non_linearity == 'logit':
		return val * (1-val)
	if non_linearity == 'tanh':
		return (1 - np.power(np.tanh(val),2))

class LSTM(object):
	"""docstring for LSTM"""
	def __init__(self, len_vec, hidden_units, steps):
		super(LSTM, self).__init__()
		self.input_,self.a,self.i,self.f,self.o,self.h,self.C,self.z = {},{},{},{},{},{},{},{}
		# self.dinput,self.da,self.di,self.df,self.do,self.dh,self.dC,self.dz = {},{},{},{},{},{},{},{}
		self.h[-1] = np.zeros((hidden_units,1))
		self.Wf,self.bf = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wi,self.bi = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wc,self.bc = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wo,self.bo = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.C[-1] = np.zeros((hidden_units,1))
		self.steps = steps
		self.len_vec = len_vec
		self.hidden_units = hidden_units
		self.dweights = np.zeros((hidden_units+len_vec,4*hidden_units))
		return

	def forward(self,x):

		for j in range(self.steps):
			self.input_[j] = np.concatenate((x[j],self.h[j-1]),axis = 0)

			#Equations for a single time step of LSTM
			self.f[j] = sigmoid(np.matmul(self.Wf.T,self.input_[j]),'logit')
			self.i[j] = sigmoid(np.matmul(self.Wi.T,self.input_[j]),'logit')
			self.a[j] = sigmoid(np.matmul(self.Wc.T,self.input_[j]),'tanh')
			self.C[j] = self.f[j]*self.C[j-1] + self.i[j]*self.a[j]  #element-wise multiplication
			self.o[j] = sigmoid(np.matmul(self.Wo.T,self.input_[j]),'logit')
			self.h[j] = self.o[j] * sigmoid(self.C[j],'tanh')
			
			#concatenate all the gates and weights for simplicity in calculation of the weight updates
			self.weights_concatenated = np.concatenate((self.Wc,self.Wi,self.Wf,self.Wo),axis = 1)
			self.z = np.matmul(self.weights_concatenated.T,self.input_[j])

		out = self.h
		self.h[-1] = self.h[j]
		self.C[-1] = self.C[j]
		return out

	def backward(self,d_above):
		
		dhnext = np.zeros_like(self.h[0])
		dc = np.zeros_like(self.C[0])
		# self.dc[self.steps],self.dh[self.steps] = dc,dh
		for t in reversed(xrange(self.steps)):
			dh = d_above[t] + dhnext
			do = dh*sigmoid(self.C[t],'tanh')
			dc += dh * self.o[t] * deriv_sigmoid(self.C[t],'tanh')
			da = dc * self.i[t]
			di = dc * self.a[t]
			df = dc * self.C[t]
			dc = dc * self.f[t]
			df_bar = df * deriv_sigmoid(self.f[t],'logit') 
			di_bar = di * deriv_sigmoid(self.i[t],'logit') 
			do_bar = do * deriv_sigmoid(self.o[t],'logit')
			da_bar = da * deriv_sigmoid(self.a[t],'tanh')
			dz = np.concatenate((df_bar,di_bar,do_bar,da_bar),axis = 0)
			dinput = np.matmul(self.weights_concatenated,dz)
			dhnext += dinput[self.len_vec:]
			self.dweights += np.matmul(dinput,dz.T)

		return dh

	def update(self):
		#Adagrad update
		pass

class Softmax(object):

	def __init__(self,len_vec, h_lstm, steps):
		self.weights = initialise_parameters(h_lstm,len_vec)[0]
		self.error_derivatives_w = np.zeros(self.weights.shape)
		# print type(self.weights)
		self.steps = steps
		self.out,self.delta_h = {},{}
		return

	def forward(self,activations_below):
		
		for t in range(self.steps):
			self.out[t] = softmax(np.matmul(self.weights.T,activations_below[t])) 
		self.activations_LSTM = activations_below
		return self.out	

	def backward(self,target):

		for t in range(self.steps):
			error_derivatives_ISM = self.out[t] - target[t]
			self.error_derivatives_w += np.matmul(self.activations_LSTM[t],error_derivatives_ISM.T)
			self.delta_h[t] = np.matmul(self.weights,error_derivatives_ISM)

		return self.delta_h

	def update(self):
		#Adagrad update
		pass
		# return