import numpy as np
from lib import *
from collections import Counter

def one_hot(char):
	vector = np.zeros(len_input)
	index = unq_chars.index(char)
	vector[index] = 1
	return vector

def train(data):
	#hyperparameters
	learning_rate = 1e-3
	hidden_nodes = 512
	seq_length = 50

	layer = LSTM(len_input,hidden_nodes)
	for char in data:
		x = one_hot(char)
		for i in range(number_steps):
			x = step(x)


 
if __name__ == '__main__':
	data = load_data()
	count = Counter(data)
	unq_chars = count.keys()
	len_input = len(unq_chars)
	train(data)