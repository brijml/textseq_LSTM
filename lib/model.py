import numpy as np

def sigmoid(z,non_linearity):
	if non_linearity == 'logit':
		return 1/(1+np.exp(-z))
	if non_linearity == 'tanh':
		return np.tanh(z)


def initialise_paramters(v,h):
	weights = np.random.randn(v,h)/np.sqrt(v*h)
	bias = 0.01 * np.random.randn(h)
	return weights,bias


def softmax(var):
	exponent = np.exp(var)
	return exponent/sum(exponent)

def deriv_sigmoid(val,non_linearity):
	if non_linearity == 'logit'
		return val * (1-val)
	if non_linearity == 'tanh'
		return (1 - np.power(np.tanh(val),2))

class LSTM(object):
	"""docstring for LSTM"""
	def __init__(self, len_vec, hidden_units):
		super(LSTM, self).__init__()
		self.ht = np.zeros(hidden_units) #65 in this case
		self.Wf,self.bf = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wi,self.bi = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wc,self.bc = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.Wo,self.bo = initialise_parameters(hidden_units+len_vec,hidden_units)
		self.C = 0

	def step(self,x):

		input_ = np.concatenate(x,self.ht)
		forget = sigmoid(np.matmul(self.Wf,input_) + self.bf,'logit')
		i = sigmoid(np.matmul(self.Wi,input_) + self.bi,'logit')
		a = sigmoid(np.matmul(self.Wc,input_) + self.bc,'tanh')
		self.C = forget * self.C + i*a
		o = sigmoid(np.matmul(self.Wo,input_) + self.bo,'logit')
		self.ht = o * sigmoid(self.C,'tanh')
		#concatenate all the gates and weights for simplicity in calculation of the weight updates
		self.weights_concatenated = np.concatenate(self.Wc,self.Wi,self.Wf,self.Wo)
		z = weights_concatenated * input_

	def backward(self,dc,dh):
		
		do = dh*sigmoid(self.C,'tanh')
		self.dc += dh * o * deriv_sigmoid(self.C,'tanh')
		da = self.dc * i
		di = self.dc * a
		df = dc * self.C
		dc_prev = self.dc * f
		df_bar = df * deriv_sigmoid(f,'logit') 
		di_bar = di * deriv_sigmoid(i,'logit') 
		do_bar = do * deriv_sigmoid(o,'logit')
		da_bar = da * deriv_sigmoid(input_,'tanh')
		dz = np.concatenate(df_bar,di_bar,do_bar,da_bar)
		dinput = self.weights_concatenated * dz
		dh_prev = d_input[len_vec:hidden_units]
		dweights = dz * dinput

	def update():
		#Adagrad update

class Softmax(object):

	def __init__(self,len_vec, h_lstm):
		self.H = len_vec
		self.fanin = h_lstm
		self.weights = initialise_parameters(self.fanin,self.H)

		return

	def forward(self,activations_below, weight_decay = None):
		self.out = softmax(np.matmul(self.weights.T,activations_below)) #+ weight_decay * np.atleast_2d(np.sum(self.weights,axis=1)).T
		self.activations_FC = activations_below
		return self.out	

	def backward(self,target):

		error_derivatives_ISM = self.out - np.atleast_2d(target).T
		self.error_derivatives_w = np.matmul(self.activations_FC,error_derivatives_ISM.T)
		delta_FC = np.matmul(self.weights,error_derivatives_ISM)

		return delta_FC

	def update(self,learning_rate,momentum):
		#Adagrad update
	return