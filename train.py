import numpy as np
from lib import *
from collections import Counter

def one_hot(char):
	vector = np.zeros((len_input,1))
	index = unq_chars.index(char)
	vector[index] = 1
	return vector

def train(data):
	#hyperparameters
	learning_rate = 1e-3
	hidden_nodes = 100
	seq_length = 25
	# hprev = np.zeros(hidden_nodes)

	layer = LSTM(len_input,hidden_nodes,seq_length)
	output = Softmax(len_input,hidden_nodes, seq_length)
	cnt = 0
	for i in range(0,len(data),seq_length):
		
		#Check for errors for 4 iterations of loop
		cnt+=1
		if cnt>4:
			break
		
		x,target = [],[]
		for j,char in enumerate(data[i:i+seq_length]):
			x.append(one_hot(char))
			target.append(one_hot(data[j+1]))

		#Forward pass
		lstm_out = layer.forward(x)
		softmax_out = output.forward(lstm_out)

		#Backward pass
		grad_softmax = output.backward(target)
		grad_lstm = layer.backward(grad_softmax)

		#Update
		layer.update()
		output.update()
 
if __name__ == '__main__':
	data = load_data()
	count = Counter(data)
	unq_chars = count.keys()
	len_input = len(unq_chars)
	train(data)