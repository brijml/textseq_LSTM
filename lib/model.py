import numpy as np

def sigmoid(z,non_linearity):
	if non_linearity == 'logit':
		return 1/(1+np.exp(-z))
	if non_linearity == 'tanh':
		return np.tanh(z)


def initialise_parameters(v,h):
	weights = np.random.randn(v,h)/np.sqrt(v*h)
	bias = 0.01 * np.random.randn(h,1)
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
	
	def __init__(self, len_vec, hidden_units, steps):
		super(LSTM, self).__init__()
		self.input_,self.a_bar,self.i,self.f,self.o,self.h,self.C,self.z = {},{},{},{},{},{},{},{}
		# self.h[-1] = h_init
		self.Wf,self.bf = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wi,self.bi = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wc,self.bc = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wo,self.bo = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.C[-1] = np.zeros((hidden_units,1))
		self.steps = steps
		self.len_vec = len_vec
		self.hidden_units = hidden_units
		self.dweights = np.zeros((hidden_units+len_vec,4*hidden_units))
		self.dbias = np.zeros((4*hidden_units,1))
		self.j,self.cache_weights,self.cache_bias = 0,0,0
		self.dh = {}
		return

	def forward(self,x):

		# print self.j
		# print x.shape,self.h[-1].shape
		self.input_[self.j] = np.concatenate((x,self.h[self.j-1]),axis = 0)

		#Equations for a single time step of LSTM
		self.f[self.j] = sigmoid(np.matmul(self.Wf.T,self.input_[self.j]) + self.bf,'logit')
		self.i[self.j] = sigmoid(np.matmul(self.Wi.T,self.input_[self.j]) + self.bi,'logit')
		self.a_bar[self.j] = np.matmul(self.Wc.T,self.input_[self.j]) + self.bc
		self.a = sigmoid(self.a_bar[self.j],'tanh')
		self.C[self.j] = self.f[self.j]*self.C[self.j-1] + self.i[self.j]*self.a  #element-wise multiplication
		self.o[self.j] = sigmoid(np.matmul(self.Wo.T,self.input_[self.j]) + self.bo,'logit')
		self.h[self.j] = self.o[self.j] * sigmoid(self.C[self.j],'tanh')
	
		#concatenate all the gates and weights for simplicity in calculation of the weight updates
		self.weights_concatenated = np.concatenate((self.Wc,self.Wi,self.Wf,self.Wo),axis = 1)
		self.bias_concatenated = np.concatenate((self.bc,self.bi,self.bf,self.bo),axis = 0)
		self.z = np.matmul(self.weights_concatenated.T,self.input_[self.j]) + self.bias_concatenated

		out = self.h[self.j]
		self.h[-1] = self.h[self.j]
		self.C[-1] = self.C[self.j]
		self.j+=1
		return out

	def backward(self,d_above):
		
		dhnext = np.zeros_like(self.h[0])
		dcnext = np.zeros_like(self.C[0])

		for t in reversed(xrange(self.steps)):
			self.dh[t] = d_above[t] + dhnext
			do = self.dh[t]*sigmoid(self.C[t],'tanh')
			dc = self.dh[t] * self.o[t] * deriv_sigmoid(self.C[t],'tanh') + dcnext
			da = dc * self.i[t]
			di = dc * self.a[t]
			df = dc * self.C[t]
			dcnext = dc * self.f[t]
			df_bar = df * deriv_sigmoid(self.f[t],'logit') 
			di_bar = di * deriv_sigmoid(self.i[t],'logit') 
			do_bar = do * deriv_sigmoid(self.o[t],'logit')
			da_bar = da * deriv_sigmoid(self.a_bar[t],'tanh')
			dz = np.concatenate((da_bar,di_bar,df_bar,do_bar),axis = 0)
			dinput = np.matmul(self.weights_concatenated,dz)
			dhnext = dinput[self.len_vec:]
			self.dweights += np.matmul(self.input_[t],dz.T)
			self.dbias += dz 
		
		for dparam in [self.dweights, self.dbias]:
			np.clip(dparam, -5, 5, out=dparam)

		return self.dh

	def update(self,learning_rate):
		#Adagrad update

		# print self.dweights
		# tagahfahf = raw_input()

		self.cache_weights += np.power(self.dweights,2)
		self.cache_bias += np.power(self.dbias,2)
		self.weights_concatenated -= (learning_rate * self.dweights)/np.sqrt(self.cache_weights + 1e-7)
		self.bias_concatenated -= (learning_rate * self.dbias)/np.sqrt(self.cache_bias + 1e-7)
		self.Wc, self.bc = self.weights_concatenated[:,0:100], self.bias_concatenated[0:100]
		self.Wf, self.bf = self.weights_concatenated[:,100:200], self.bias_concatenated[100:200]
		self.Wi, self.bi = self.weights_concatenated[:,200:300], self.bias_concatenated[200:300]
		self.Wo, self.bo = self.weights_concatenated[:,300:400], self.bias_concatenated[300:400]
		return
	
	def reset(self):
		self.j = 0
		return

class Softmax(object):

	def __init__(self,len_vec, h_lstm, steps):
		self.weights,self.bias = initialise_parameters(h_lstm,len_vec)
		self.error_derivatives_w = np.zeros_like(self.weights)
		self.error_derivatives_b = np.zeros_like(self.bias)
		self.steps = steps
		self.t,self.cache_weights,self.cache_bias = 0,0,0
		self.out,self.delta_h = {},{}
		return

	def forward(self,activations_below):
		
		self.out[self.t] = softmax(np.matmul(self.weights.T,activations_below) + self.bias) 
		self.activations_LSTM = activations_below
		out = self.out[self.t]
		self.t+=1
		return out	

	def backward(self,target):

		for t in range(self.steps):
			error_derivatives_ISM = self.out[t] - target[t]
			self.error_derivatives_w += np.matmul(self.activations_LSTM[t],error_derivatives_ISM.T)
			self.error_derivatives_b += error_derivatives_ISM
			self.delta_h[t] = np.matmul(self.weights,error_derivatives_ISM)
		
		for dparam in [self.error_derivatives_w, self.error_derivatives_w]:
			np.clip(dparam, -5, 5, out=dparam)


		return self.delta_h

	def update(self,learning_rate):
		
		#Adagrad update
		self.cache_weights += np.power(self.error_derivatives_w,2)
		self.cache_bias += np.power(self.error_derivatives_b,2)
		self.weights -= (learning_rate * self.error_derivatives_w)/np.sqrt(self.cache_weights + 1e-7)
		self.bias -= (learning_rate * self.error_derivatives_b)/np.sqrt(self.cache_bias + 1e-7)
		return

	def reset(self):
		self.t = 0
		return

class RNN(object):
	
	def __init__(self, len_vec, hidden_units, steps):
		super(RNN, self).__init__()
		self.h,self.input_ = {},{}
		# self.h[-1] = h_init
		self.Wxh = initialise_parameters(len_vec,hidden_units)[0]
		self.Whh,self.bh = initialise_parameters(hidden_units,hidden_units)
		self.steps = steps
		self.dWxh = np.zeros_like(self.Wxh)
		self.dWhh = np.zeros_like(self.Whh)
		self.dbh = np.zeros_like(self.bh)
		self.j = 0
		return

	def forward(self,x):

		#Equations for a single time step of RNN
		self.input_[self.j] = x
		self.h[self.j] = sigmoid(np.matmul(self.Whh.T,self.h[self.j-1]) + np.matmul(self.Wxh.T,x) + self.bh,'tanh')
		out = self.h[self.j]
		self.h[-1] = self.h[self.j]
		self.j+=1
		return out

	def backward(self,d_above):
		
		dhnext = np.zeros_like(self.h[0])

		for t in reversed(xrange(self.steps)):
			dh = d_above[t] + dhnext
			dhraw = (1 - self.h[t]*self.h[t]) * dh
			# print dhraw.shape,self.input_[t].shape
			self.dbh += dhraw
			self.dWxh += np.matmul(self.input_[t], dhraw.T)
			self.dWhh += np.dot(dhraw, self.h[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)		

		for dparam in [self.dWxh,self.dWhh,self.dbh]:
			np.clip(dparam, -5, 5, out=dparam)

		return dh

	def update(self,learning_rate):
		#Adagrad update

		self.Wxh -= (learning_rate * self.dWxh)
		self.Whh -= (learning_rate * self.dWhh)
		self.bh -= (learning_rate * self.dbh)
		return
	
	def reset(self):
		self.j = 0
		return