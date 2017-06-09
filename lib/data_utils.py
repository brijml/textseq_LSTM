import nltk,os
import numpy as np
import cPickle

path = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(path,'input.txt')
save_path = os.path.join(path, 'network_parameters')
temp_path = os.path.join(path, 'temp_parameters')

def load_data():
	characters = []
	# with open(data_file,'rb') as f:
	data = open(data_file, 'r').read()

	return data


def dump_parameters(lstm1_weights,lstm1_bias,lstm2_weights,lstm2_bias,softmax_weights,softmax_bias,iters):

	dict_ = {"lstm1_weights":lstm1_weights,'lstm1_bias':lstm1_bias,\
				"lstm2_weights":lstm2_weights,'lstm2_bias':lstm2_bias,\
				"softmax_weights":softmax_weights,'softmax_bias':softmax_bias}
	filename = os.path.join(save_path, 'parameters_'+str(iters)+'.pickle')
	with open(filename, 'wb"') as output_file:
		cPickle.dump(dict_, output_file)
	print 'parameters saved for : ', filename

	return

def dump_parameters_temp(lstm1_weights,lstm1_bias,lstm2_weights,lstm2_bias,softmax_weights,softmax_bias,iters,cnt):

	dict_ = {"lstm1_weights":lstm1_weights,'lstm1_bias':lstm1_bias,\
				"lstm2_weights":lstm2_weights,'lstm2_bias':lstm2_bias,\
				"softmax_weights":softmax_weights,'softmax_bias':softmax_bias}
	filename = os.path.join(temp_path, 'parameters_'+str(iters)+'_'+str(cnt)+'.pickle')
	with open(filename, 'wb"') as output_file:
		cPickle.dump(dict_, output_file)
	print 'parameters saved for : ', filename

	return

def load_model(file):

	filename = os.path.join(save_path,file)
	with open(filename,"rb") as f:
		parameters = cPickle.load(f)

	return parameters

def load_model_temp(file):

	filename = os.path.join(temp_path,file)
	with open(filename,"rb") as f:
		parameters = cPickle.load(f)

	return parameters
	
if __name__ == '__main__':
	char = load_data()