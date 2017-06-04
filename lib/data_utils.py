import nltk,os
import numpy as np

path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'input.txt')

def load_data():
	characters = []
	with open(path,'rb') as f:
		for line in f.readlines():
			characters.extend([i for i in line])

	return characters

if __name__ == '__main__':
	char = load_data()