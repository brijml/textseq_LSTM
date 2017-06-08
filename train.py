import numpy as np
from lib import *
import sys
import matplotlib.pyplot as plt

def one_hot(char):
	vector = np.zeros((len_input,1))
	index = unq_chars.index(char)
	vector[index] = 1
	return vector

def sample(x,len_text):

	ixes = []
	x = one_hot(x)
	for i in range(len_text):
		lstm_out = layer.forward(x)
		p = output.forward(lstm_out)
		ix = np.random.choice(range(len_input), p=p.ravel())
		x = one_hot(unq_chars[ix])
		ixes.append(ix)
	
	layer.reset()
	output.reset()
	
	return ixes


def train(data,load_pretrained = False):
	global layer,output
	#hyperparameters
	learning_rate = 1e-5
	hidden_nodes = 100
	seq_length = 25
	
	#Initial value for the LSTM
	h_init = np.zeros((hidden_nodes,1))

	layer = LSTM(len_input,hidden_nodes,seq_length,h_init)
	output = Softmax(len_input,hidden_nodes, seq_length)
	
	if load_pretrained:
		parameters = load_model("parameters_5.pickle")
		layer.weights_concatenated = parameters['lstm_weights']
		layer.bias_concatenated = parameters['lstm_bias']
		layer.Wc, layer.bc = layer.weights_concatenated[:,0:100], layer.bias_concatenated[0:100]
		layer.Wf, layer.bf = layer.weights_concatenated[:,100:200], layer.bias_concatenated[100:200]
		layer.Wi, layer.bi = layer.weights_concatenated[:,200:300], layer.bias_concatenated[200:300]
		layer.Wo, layer.bo = layer.weights_concatenated[:,300:400], layer.bias_concatenated[300:400]
		output.weights = parameters['softmax_weights']
		output.bias = parameters['softmax_bias']

	cnt,iters = 1,6
	error,val = [],[]
	
	while True:
		print "Epoch no", iters
		
		for i in range(0,len(data)-seq_length-1,seq_length):
			
			x,target = [],[]
			for j,char in enumerate(data[i:i+seq_length]):
				x = one_hot(char)
				target = one_hot(data[j+1])

				#Forward pass
				lstm_out = layer.forward(x)
				softmax_out = output.forward(lstm_out)

			#Backward pass
			grad_softmax = output.backward(target)
			grad_lstm = layer.backward(grad_softmax)

			#Update
			layer.update(learning_rate)
			output.update(learning_rate)

			layer.reset()
			output.reset()

			#Sample text every now and then to see what lstm is learning
			if cnt%100 == 0:

				indices = sample(data[0],200)
				text = ""
				for idx in indices:
					text = text + unq_chars[idx]

				print '-------\n',text,'\n-------'

			cnt+=1 #iteration counter 	

		dump_parameters(layer.weights_concatenated,layer.bias_concatenated,output.weights,output.bias,iters)
		iters+=1

if __name__ == '__main__':
	data = load_data()
	print len(data)
	unq_chars = list(set(data))
	len_input = len(unq_chars)
	train(data,load_pretrained = True)