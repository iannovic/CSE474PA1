import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  #your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    #Your code here
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    #Get data for training/validation
    train0 = mat.get('train0')
    train0_size = train0.shape[0]
    train_label0 = np.zeros(train0_size)
    for i in range(train0_size):
        train_label0[i] = 0
    
    train1 = mat.get('train1')
    train1_size = train1.shape[0]
    train_label1 = np.zeros(train1_size)
    for i in range(train1_size):
        train_label1[i] = 1

    train2 = mat.get('train2')
    train2_size = train2.shape[0]
    train_label2 = np.zeros(train2_size)
    for i in range(train2_size):
        train_label2[i] = 2
    
    train3 = mat.get('train3')
    train3_size = train3.shape[0]
    train_label3 = np.zeros(train3_size)
    for i in range(train3_size):
        train_label3[i] = 3

    train4 = mat.get('train4')
    train4_size = train4.shape[0]
    train_label4 = np.zeros(train4_size)
    for i in range(train4_size):
        train_label4[i] = 4
    
    train5 = mat.get('train5')
    train5_size = train5.shape[0]
    train_label5 = np.zeros(train5_size)
    for i in range(train5_size):
        train_label5[i] = 5

    train6 = mat.get('train6')
    train6_size = train6.shape[0]
    train_label6 = np.zeros(train6_size)
    for i in range(train6_size):
        train_label6[i] = 6

    train7 = mat.get('train7')
    train7_size = train7.shape[0]
    train_label7 = np.zeros(train7_size)
    for i in range(train7_size):
        train_label7[i] = 7

    train8 = mat.get('train8')
    train8_size = train8.shape[0]
    train_label8 = np.zeros(train8_size)
    for i in range(train8_size):
        train_label8[i] = 8

    train9 = mat.get('train9')
    train9_size = train9.shape[0]
    train_label9 = np.zeros(train9_size)
    for i in range(train9_size):
        train_label9[i] = 9

    train_vstack_1 = np.vstack((train0, train1))
    train_vstack_2 = np.vstack((train_vstack_1, train2))
    train_vstack_3 = np.vstack((train_vstack_2, train3))
    train_vstack_4 = np.vstack((train_vstack_3, train4))
    train_vstack_5 = np.vstack((train_vstack_4, train5))
    train_vstack_6 = np.vstack((train_vstack_5, train6))
    train_vstack_7 = np.vstack((train_vstack_6, train7))
    train_vstack_8 = np.vstack((train_vstack_7, train8))
    train_vstack_9 = np.vstack((train_vstack_8, train9))

    print train_label0

    train_concat_1 = np.concatenate((train_label0, train_label1))
    train_concat_2 = np.concatenate((train_concat_1, train_label2))
    train_concat_3 = np.concatenate((train_concat_2, train_label3))
    train_concat_4 = np.concatenate((train_concat_3, train_label4))
    train_concat_5 = np.concatenate((train_concat_4, train_label5))
    train_concat_6 = np.concatenate((train_concat_5, train_label6))
    train_concat_7 = np.concatenate((train_concat_6, train_label7))
    train_concat_8 = np.concatenate((train_concat_7, train_label8))
    train_concat_9 = np.concatenate((train_concat_8, train_label9))

    #np.concatenate((train_label0, train_label3))
    #np.concatenate((train_label0, train_label4))
    #np.concatenate((train_label0, train_label5))
    #np.concatenate((train_label0, train_label6))
    #np.concatenate((train_label0, train_label7))
    #np.concatenate((train_label0, train_label8))
    #np.concatenate((train_label0, train_label9))

    #Convert all to double and normailze so that it is between 0 and 1
    for k in range(train0.shape[0]):
        for t in range(784):
            train0[k][t] = np.double(train0[k][t])
            train0[k][t] = train0[k][t]/256

    print train_concat_9
    print train_vstack_9.shape[0]

    #Get data for testing
    test0 = mat.get('test0')
    test0_size = test0.shape[0]
    test_label0 = np.zeros(test0_size)
    for i in range(test0_size):
        test_label0[i] = 0

    test1 = mat.get('test1')
    test1_size = test1.shape[0]
    test_label1 = np.zeros(test1_size)
    for i in range(test1_size):
        test_label1[i] = 1

    test2 = mat.get('test2')
    test2_size = test2.shape[0]
    test_label2 = np.zeros(test2_size)
    for i in range(test2_size):
        test_label2[i] = 2

    test3 = mat.get('test3')
    test3_size = test3.shape[0]
    test_label3 = np.zeros(test3_size)
    for i in range(test3_size):
        test_label3[i] = 3

    test4 = mat.get('test4')
    test4_size = test4.shape[0]
    test_label4 = np.zeros(test4_size)
    for i in range(test4_size):
        test_label4[i] = 4

    test5 = mat.get('test5')
    test5_size = test5.shape[0]
    test_label5 = np.zeros(test5_size)
    for i in range(test5_size):
        test_label5[i] = 5

    test6 = mat.get('test6')
    test6_size = test6.shape[0]
    test_label6 = np.zeros(test6_size)
    for i in range(test6_size):
        test_label6[i] = 6

    test7 = mat.get('test7')
    test7_size = test7.shape[0]
    test_label7 = np.zeros(test7_size)
    for i in range(test7_size):
        test_label7[i] = 7

    test8 = mat.get('test8')
    test8_size = test8.shape[0]
    test_label8 = np.zeros(test8_size)
    for i in range(test8_size):
        test_label8[i] = 8

    test9 = mat.get('train9')
    test9_size = test9.shape[0]
    test_label9 = np.zeros(test9_size)
    for i in range(test9_size):
        test_label9[i] = 9

    test_vstack_1 = np.vstack((test0, test1))
    test_vstack_2 = np.vstack((test_vstack_1, test2))
    test_vstack_3 = np.vstack((test_vstack_2, test3))
    test_vstack_4 = np.vstack((test_vstack_3, test4))
    test_vstack_5 = np.vstack((test_vstack_4, test5))
    test_vstack_6 = np.vstack((test_vstack_5, test6))
    test_vstack_7 = np.vstack((test_vstack_6, test7))
    test_vstack_8 = np.vstack((test_vstack_7, test8))
    test_vstack_9 = np.vstack((test_vstack_8, test9))

    test_concat_1 = np.concatenate((test_label0, test_label1))
    test_concat_2 = np.concatenate((test_concat_1, test_label2))
    test_concat_3 = np.concatenate((test_concat_2, test_label3))
    test_concat_4 = np.concatenate((test_concat_3, test_label4))
    test_concat_5 = np.concatenate((test_concat_4, test_label5))
    test_concat_6 = np.concatenate((test_concat_5, test_label6))
    test_concat_7 = np.concatenate((test_concat_6, test_label7))
    test_concat_8 = np.concatenate((test_concat_7, test_label8))
    test_concat_9 = np.concatenate((test_concat_8, test_label9))

    #np.concatenate((test0, test1))
    #np.concatenate((test0, test2))
    #np.concatenate((test0, test3))
    #np.concatenate((test0, test4))
    #np.concatenate((test0, test5))
    #np.concatenate((test0, test6))
    #np.concatenate((test0, test7))
    #np.concatenate((test0, test8))
    #np.concatenate((test0, test9))
    
    #np.concatenate((test_label0, test_label1))
    #np.concatenate((test_label0, test_label2))
    #np.concatenate((test_label0, test_label3))
    #np.concatenate((test_label0, test_label4))
    #np.concatenate((test_label0, test_label5))
    #np.concatenate((test_label0, test_label6))
    #np.concatenate((test_label0, test_label7))
    #np.concatenate((test_label0, test_label8))
    #np.concatenate((test_label0, test_label9))

    #Convert all to double and normailze so that it is between 0 and 1
    for k in range(test_concat_9.shape[0]):
        for t in range(784):
            test_vstack_9[k][t] = np.double(test_vstack_9[k][t])
            test_vstack_9[k][t] = test_vstack_9[k][t]/255

    #Randomly select 10,000 for validation

    a = range(train_vstack_9.shape[0])
    aperm = np.random.permutation(a)

    validation_data = train_vstack_9[aperm[0:10000],:]
    train_data = train_vstack_9[aperm[10000:],:]

    validation_label = train_concat_9[aperm[0:10000]]
    train_label = train_concat_9[aperm[10000:]]
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #
    
    #create target vector class to compare during back propogation
    
    target_class = 	([1,0,0,0,0,0,0,0,0,0]    #0 target class
    			,[0,1,0,0,0,0,0,0,0,0]    #1
    			,[0,0,1,0,0,0,0,0,0,0]	  #2
    			,[0,0,0,1,0,0,0,0,0,0]    #3
    			,[0,0,0,0,1,0,0,0,0,0]    #4
    			,[0,0,0,0,0,1,0,0,0,0]    #5
    			,[0,0,0,0,0,0,1,0,0,0]    #6
    			,[0,0,0,0,0,0,0,1,0,0]    #7
    			,[0,0,0,0,0,0,0,0,1,0]    #8
    			,[0,0,0,0,0,0,0,0,0,1]);  #9 
    			
    #end of target vector init
    
    for i in (0,5000):
        input_vectors_1 = np.zeros((n_input,n_hidden));
        input_vectors_2 = np.zeros((n_hidden,n_class));
        output_i = np.zeros(n_class);

        for d in (0,n_input):
            for m in (0,n_hidden):
                input_vectors_1[d][m] = w1[d][m] * train_data[i][d];

        for m in (0,n_hidden):
            net_m = 0;
            for d in (0,n_input):
                net_m += input_vectors_1[d][m];
            for l in (0,n_class):
                input_vectors_2[m][l] = net_m;

        for l in (0,n_class):
            net_l = 0;
            for m in (0,hidden):
                net_l += input_vectors_2[m][l] * w2[m][l]; #SIGMOID THIS LINE
            output_i[l] = net_l; #SIGMOID THIS LINE


	#for each weight path m,l update the weight based on the output
	for m in (0,n_hidden):
		for l in (0,n_class):
			greek_squiggly_letter = o[l] - target_class[current_training_label][l];
			zee_jay = input_vectors_2[m][l]
			w2[m][l] = w2[m][l] - learning_rate * greek_squiggly_letter * zee_jay

    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#Temp_VARS
NUM_INPUT_NODES = 1;

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[NUM_INPUT_NODES]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
