import nltk,os
import numpy as np
import cPickle

path = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(path,'input.txt')
save_path = os.path.join(path, 'network_parameters')

def load_data():
	characters = []
	with open(data_file,'rb') as f:
		for line in f.readlines():
			characters.extend([i for i in line])

	return characters


def dump_parameters(lstm,softmax,iters):

	dict_ = {"lstm":lstm,"softmax":softmax}
	filename = os.path.join(save_path, 'parameters_'+str(iters)+'.pickle')
	with open(filename, 'wb"') as output_file:
		cPickle.dump(dict_, output_file)
	print 'parameters saved for : ', filename

	return

if __name__ == '__main__':
	char = load_data()