import numpy as np

# Sigmoid Function
def sig_moid(z):
	return 1 / (1 + np.exp(-z))



def init_param(input_features, number_of_hidden_layers, output_features):
	input_w = np.random.randn(number_of_hidden_layers, input_features)
	input_w_ = np.random.randn(output_features, number_of_hidden_layers)
	input_b = np.zeros((number_of_hidden_layers, 1))
	input_b_ = np.zeros((output_features, 1))
	
	param = {"input_w" : input_w, "input_b": input_b,
				"input_w_" : input_w_, "input_b_": input_b_}
	return param

# Forward Propagation
def forward_prop(inputs, outputs, param):
	m = inputs.shape[1]
	input_w = param["input_w"]
	input_w_ = param["input_w_"]
	input_b = param["input_b"]
	input_b_ = param["input_b_"]

	input_z = np.dot(input_w, inputs) + input_b
	input_a = sig_moid(input_z)
	input_z_ = np.dot(input_w_, input_a) + input_b_
	input_a_ = sig_moid(input_z_)

	ca = (input_z, input_a, input_w, input_b, input_z_, input_a_, input_w_, input_b_)
	logprobs = np.multiply(np.log(input_a_), outputs) + np.multiply(np.log(1 - input_a_), (1 - outputs))
	cost = -np.sum(logprobs) / m
	return cost, ca, input_a_

# Backward Propagation
def back_prop(inputs, outputs, ca):
	m = inputs.shape[1]
	(input_z, input_a, input_w, input_b, input_z_, input_a_, input_w_, input_b_) = ca
	
	_input_z_ = input_a_ - outputs
	inputs_w_ = np.dot(_input_z_, input_a.T) / m
	inputs_b_ = np.sum(_input_z_, axis = 1, keepdims = True)
	
	inputs_a = np.dot(input_w_.T, _input_z_)
	input_z_ = np.multiply(inputs_a, input_a * (1- input_a))
	inputs_w = np.dot(input_z_, inputs.T) / m
	inputs_b = np.sum(input_z_, axis = 1, keepdims = True) / m
	
	grad = {"_input_z_": _input_z_, "inputs_w_": inputs_w_, "inputs_b_": inputs_b_,
				"input_z_": input_z_, "inputs_w": inputs_w, "inputs_b": inputs_b}
	return grad

# Updating the weights
def updating_param(param, grad, learn_rate):
	param["input_w"] = param["input_w"] - learn_rate * grad["inputs_w"]
	param["input_w_"] = param["input_w_"] - learn_rate * grad["inputs_w_"]
	param["input_b"] = param["input_b"] - learn_rate * grad["inputs_b"]
	param["input_b_"] = param["input_b_"] - learn_rate * grad["inputs_b_"]
	return param

def train_model(inputs,outputs):
     

    # Defining model param
    number_of_hidden_layers = 3 
    input_features = inputs.shape[0]
    output_features = outputs.shape[0] 
    param = init_param(input_features, number_of_hidden_layers, output_features)
    
    # epochs = 100000
    # learn_rate = 0.01
    # losses = np.zeros((epochs, 1))

    # for i in range(epochs):
    #     losses[i, 0], ca, input_a_ = forward_prop(inputs, outputs, param)
    #     grad = back_prop(inputs, outputs, ca)
    #     param = updating_param(param, grad, learn_rate)
    
    return param
    
def testing_model(inputs,param):
    cost, _, input_a_ = forward_prop(inputs, outputs, param)
    prediction = (input_a_ > 0.5) * 1.0
    return prediction



if __name__=="__main__":

    
    inputs = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) 
    outputs = np.array([[1, 0, 0, 1]])

    print("The actual values are: ", outputs.tolist()[0])


    #training the model
    param_=train_model(inputs,outputs)

    for i in param_:
        print(i,"\n",param_[i])
        
    #testing our model
    # pred=testing_model(inputs,param_)

    # print("The predicted values are: ",pred)
