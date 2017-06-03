import nltk
import numpy as np

def load_data():
	characters = []
	with open('input.txt','rb') as f:
		for line in f.readlines():
			characters.extend([i for i in line])

	return characters

if __name__ == '__main__':
	char = load_data()