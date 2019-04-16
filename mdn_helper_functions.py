import numpy as np
from keras import backend as K
from keras.layers import Dense, Input,concatenate
import math

"""
In this file, you will implement the mixture density network helper functions.

-----------------------------------------------------------------------------------
Please review Bishop's Mixture Density Network technical paper, especially Section 3.
Many of the functions require information from the MDN paper.

-----------------------------------------------------------------------------------
For many computations below, you will need to use "element-wise" addition,
subtraction, multiplication, division, and square on matrices. When you are 
operating on the output of a Keras neural nework, the +, -, *, /, and ** operators
perform element-wise operations. For other operations, you will need to compute sums
and means. In those cases, you must use the Keras 'backend' functions. Please 
review the Keras backend webpage for the list of operations.

Keras Backend: https://keras.io/backend/

-----------------------------------------------------------------------------------
To use the functions you completed here in other files, you can import this file
as a module by adding the following line to the top of your other Python source
files:
from mdn_helper_functions import * 
"""

def add_univariate_mixture_layer(previous_layer, nb_components):
    """
    This functions adds a univariate mixture layer onto :previous_layer:.
    """
    # Start coding here.
    # Hint: Review Section 3 of Bishop's paper to understand how to implement
    #       the mixture layer. You will create three new layers: one for the 
    #       mixture coefficients, one for the means, and one for the standard 
    #       deviations.
    #print "# of components: %d" % nb_components
    #print "# of inputs: %d" % previous_layer.get_shape()[1]

    #mdn_input=Input(shape=(previous_layer.get_shape()[1],), dtype='float32', name='mdn_input')
    mix_coeffs = Dense(nb_components,activation=K.softmax,name="mdn_mix_coeffs")(previous_layer)
    means = Dense(nb_components,name="mdn_means")(previous_layer)
    stdvs = Dense(nb_components,activation=K.exp,name="mdn_stdvs")(previous_layer)
    
    # Hint: Here, you need to concatenate the layers you created in this  
    #       function into one output layer. You must use the concatenate layer 
    #       in Keras to concatenate the three layers.
    mdn_output=concatenate([mix_coeffs,means,stdvs])

    # Return the concatenated output layer here.
    return mdn_output
    
#end def

def separate_mixture_matrix_into_parameters(mdn_output_matrix, nb_components):
    """
    This function takes the output matrix of a mixture density network and separates 
    the output matrix into three separate individual matrices: the mixture coefficient 
    matrix, the means matrix, and the standard deviations matrix.
    """

    # Write the code separates the MDN output matrix 'mdn_output_matrix' into three 
    # separate matrices here.
    # Hint: the output layer of your MDN concatenated the mixture coefficients, means, 
    #       and standard deviations into one matrix. therefore, you must use Python 
    #       splicing to separate the "mdn_output_matrix" into three individual matrices.
    #       "mdn_output_matrix" will be a two-dimensional matrix, so you must splice
    #       either the rows or the columns. you must also determine where to splice.
    #       Note, where to splice has something to do with the parameter nb_components.
    mix_coeffs_matrix = mdn_output_matrix[:,:nb_components]
    means_matrix = mdn_output_matrix[:,nb_components:2*nb_components]
    stdvs_matrix = mdn_output_matrix[:,2*nb_components:]
    
    # Return statement here. You must return the three matrices as a tuple and in
    # the following order: mix_coeffs_matrix, means_matrix, stdvs_matrix.
    return mix_coeffs_matrix, means_matrix, stdvs_matrix
    
#end def

def compute_gaussian_kernel_probability_matrix(target_array, means_matrix, stdvs_matrix):
    """
    This functions computes the probability of each Gaussian kernel using the means_matrix 
    the stdvs_matrix.
    """
    # Convert the 'target' array into a row vector
    target_matrix = K.reshape(target_array, [K.shape(target_array)[0], 1])

    # 'Tile' the target row vector to match the width of the means matrix
    target_matrix = K.tile(target_matrix, [1, K.shape(means_matrix)[1]])

    # Write the code that will compute the probability of 'target_matrix' occuring using
    # the Gaussian function.
    # Hint: Here, you are writing code that computes the mathematical formula for a 
    #       Gaussian function. It is easier to compute the Gaussian function in
    #       multiple steps.
    #       You can use the following links as references to the Gaussian function:
    #       - https://en.wikipedia.org/wiki/Gaussian_function
    #       - http://mathworld.wolfram.com/NormalDistribution.html
    
    # -- Compute the exponential portion of the Gaussian function here.
    #print "Target matrix shape: ",target_matrix.get_shape()
    #print "Means matrix shape: ", means_matrix.get_shape()
    #print "Stdvs matrix shape: ",stdvs_matrix.get_shape()
    exp = K.exp(-((target_matrix - means_matrix)**2) / (2*(stdvs_matrix**2)))
    
    # -- Compute the Gaussian function's normalizer here.
    normalizer=1/(math.sqrt(2*math.pi) * stdvs_matrix)
    
    
    # -- Compute the product of the Gaussian function's exponential and the
    #    normalizer here.
    kernel_probability_matrix=exp*normalizer
    
    # Return the result of the product here.
    return kernel_probability_matrix

#end def

def compute_total_probability_vector(mix_coeff_matrix, kernel_probability_matrix):
    """
    Computes the total, weighted probability vector using the mixture coefficient matrix and the kernel probability matrix.
    """
    # Start writing code here. The computation for the total probability vector can be
    # written in one line!
    total_probability_vector=K.sum(mix_coeff_matrix*kernel_probability_matrix,axis=1, keepdims=True)
    
    # Return statement here.
    return total_probability_vector

#end def

def negative_log_likelihood_loss(nb_components, space_holder_param=None):
    """
    This function serves as a wrapper for the negative log likelihood loss function.
    """
    # To use this function as your loss function, you must specify it in the Keras model compile
    # method.
    #
    # E.g. model.compile(..., loss=negative_log_likelihood_loss(4), ...)
    # In this example, 4 specifies the number of components your mixture density network uses.

    def loss_fnc(target_groundtruth, target_predicted):
        """
        Computes the negative log likelihood loss.
        """
        # All loss function code should be written inside here. Here, you use 'target_predicted' 
        #     (which is the output from the MDN), 'target_groundtruth' (which is the true target
        #     values), and 'nb_components' with the helper functions you defined above to
        #     compute the total probability vector. You should then use the total probability
        #     vector to compute the loss. Please note that the total probability can also be
        #     referred to as the conditional density of the complete target vector in the MDN
        #     paper.
        mix_coeffs_matrix, means_matrix, stdvs_matrix=separate_mixture_matrix_into_parameters(target_predicted,nb_components)
        kernel_probability_matrix=compute_gaussian_kernel_probability_matrix(target_groundtruth,means_matrix,stdvs_matrix)
        total_probability_vector=compute_total_probability_vector(mix_coeffs_matrix,kernel_probability_matrix)
        print total_probability_vector
        loss_input = K.sum(total_probability_vector, axis=1, keepdims=True)
        loss = -K.log(loss_input)


        # Return the log loss here.
        return K.mean(loss)

    #end def
    return loss_fnc
#end def

def compute_mixture_total_mean_variance(mix_coeff_matrix, means_matrix, stdvs_matrix):
    """
    Computes the total mean and the total variance of the entire distribution given a set of mixture coefficients, means, and standard deviations.
    """
    # To compute the total mean and the total variance, you must use NumPy functions. Please note 
    # that you must use explicit function operators such as multiply, add, sum, subtract,
    # and so on. You must NOT use multiplication (*) operator. Please review how to use these
    # functions on the NumPy website.
    # Please review Bishop's paper to understand how to compute the total mean and total variance.
    # The total mean is also known as the conditional average of the target data.

    # Compute the total mean here.
    #print "||",mix_coeff_matrix.shape,means_matrix.shape
    total_mean=np.sum(np.multiply(mix_coeff_matrix,means_matrix),axis=1)
    #print "||",total_mean.shape

    # Compute the total variance here.
    values_matrix=stdvs_matrix**2 + np.subtract(means_matrix,np.sum(np.multiply(mix_coeff_matrix,means_matrix),axis=1,keepdims=True))**2
    total_var=np.sum(np.multiply(mix_coeff_matrix,values_matrix),axis=1)

    # Return the total mean and total variance as a tuple and in the following order: mean, variance
    return total_mean, total_var

#end def

def compute_max_component_mean_variance(mix_coeff_matrix, means_matrix, stdvs_matrix):
    """
    Returns the mean and standard deviation from the component with the highest mixture coefficient.
    """
    # To get the mean and standard deviation from the component with the highest mixture
    # coefficient, you must use the NumPy functions. Please note that you must use explicit 
    # function operators such as multiply, add, sum, subtract, and so on. You must NOT use
    # multiplication (*) operator. Please review how to use these functions on the NumPy 
    # website. Please review Bishop's paper to understand how to compute the max component 
    # mean and standard deviation.

    # Find the component with the largest mixture coefficient.
    max_comp_index=np.argmax(mix_coeff_matrix,axis=1)
    #print max_comp_index
    # Compute the mean and standard deviation.
    max_comp_mean=means_matrix[np.arange(len(means_matrix)),max_comp_index]
    #print max_comp_mean
    max_comp_stdv=stdvs_matrix[np.arange(len(stdvs_matrix)),max_comp_index]


    # Return the computed mean and standard deviation.
    return max_comp_mean,max_comp_stdv

#end def
