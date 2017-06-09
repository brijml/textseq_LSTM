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
		lstm1_out = layer1.forward(x)
		lstm2_out = layer2.forward(lstm1_out)
		p = output.forward(lstm2_out)
		ix = np.random.choice(range(len_input), p=p.ravel())
		x = one_hot(unq_chars[ix])
		ixes.append(ix)
	
	layer1.reset()
	layer2.reset()
	output.reset()
	# print p
	return ixes


def train(data,load_pretrained = False):
	global layer1,layer2,output
	#hyperparameters
	learning_rate = 1e-5
	hidden_nodes = 100
	seq_length = 25
	
	#Initial value for the LSTM
	h_init = np.zeros((hidden_nodes,1))

	layer1 = LSTM(len_input,hidden_nodes,seq_length)
	layer2 = LSTM(hidden_nodes,hidden_nodes,seq_length)
	output = Softmax(len_input,hidden_nodes, seq_length)
	
	plt.ion()
	if load_pretrained:
		parameters = load_model("parameters_3.pickle")
		layer1.weights_concatenated = parameters['lstm1_weights']
		layer1.bias_concatenated = parameters['lstm1_bias']
		layer1.Wc, layer1.bc = layer1.weights_concatenated[:,0:100], layer1.bias_concatenated[0:100]
		layer1.Wf, layer1.bf = layer1.weights_concatenated[:,100:200], layer1.bias_concatenated[100:200]
		layer1.Wi, layer1.bi = layer1.weights_concatenated[:,200:300], layer1.bias_concatenated[200:300]
		layer1.Wo, layer1.bo = layer1.weights_concatenated[:,300:400], layer1.bias_concatenated[300:400]
		layer2.weights_concatenated = parameters['lstm2_weights']
		layer2.bias_concatenated = parameters['lstm2_bias']
		layer2.Wc, layer2.bc = layer2.weights_concatenated[:,0:100], layer2.bias_concatenated[0:100]
		layer2.Wf, layer2.bf = layer2.weights_concatenated[:,100:200], layer2.bias_concatenated[100:200]
		layer2.Wi, layer2.bi = layer2.weights_concatenated[:,200:300], layer2.bias_concatenated[200:300]
		layer2.Wo, layer2.bo = layer2.weights_concatenated[:,300:400], layer2.bias_concatenated[300:400]

		output.weights = parameters['softmax_weights']
		output.bias = parameters['softmax_bias']

	cnt,iters = 1,4
	error,val = [],[]
	
	while True:
		print "Epoch no", iters
		layer1.h[-1] = h_init
		layer2.h[-1] = h_init
		for i in range(0,len(data)-seq_length-1,seq_length):
			
			x,target = [],[]
			for j,char in enumerate(data[i:i+seq_length]):
				x = one_hot(char)
				target = one_hot(data[j+1])

				#Forward pass
				lstm1_out = layer1.forward(x)
				lstm2_out = layer2.forward(lstm1_out)
				softmax_out = output.forward(lstm2_out)
				error.append(np.sum(abs(target - softmax_out)))

			#Backward pass
			grad_softmax = output.backward(target)
			grad_lstm2 = layer2.backward(grad_softmax)
			grad_lstm1 = layer1.backward(grad_lstm2)

			#Update
			layer1.update(learning_rate)
			layer2.update(learning_rate)
			output.update(learning_rate)

			layer1.reset()
			layer2.reset()
			output.reset()

			#Sample text every now and then to see what lstm is learning
		
			if cnt%5000 == 0:
				indices = sample(data[i],200)
				text = ""
				for idx in indices:
					text = text + unq_chars[idx]

				print '-------\n',text,'\n-------'
				dump_parameters_temp(layer1.weights_concatenated,layer1.bias_concatenated,\
						layer2.weights_concatenated,layer2.bias_concatenated,\
						output.weights,output.bias,iters,cnt)
			cnt+=1 #iteration counter 	

		dump_parameters(layer1.weights_concatenated,layer1.bias_concatenated,\
						layer2.weights_concatenated,layer2.bias_concatenated,\
						output.weights,output.bias,iters)
		iters+=1

if __name__ == '__main__':
	data = load_data()
	print len(data)
	unq_chars = list(set(data))
	len_input = len(unq_chars)
	train(data,load_pretrained = True)