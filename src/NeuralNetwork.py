'''
    Neural Network Class
'''

# neural network class definition
import numpy
import scipy.special


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''
        initialise the neural network
        :param inputnodes:
        :param hiddennodes:
        :param outputnodes:
        :param learningrate:
        '''
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        # link weights
        # wih -- weight between input and hidden
        # who -- weight between hidden and output
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    def train(self, inputs_list, targets_list):
        '''
        train the neural network
        :param inputs_list: [1,2,3,4,5,6...]
        :param targets_list: [1,2,3,4,5,6...]
        :return: Null
        '''
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer -- outputs of hidden
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error using BP
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights between the hidden and output layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        '''
        query the neural network
        :param inputs_list: [1,2,3,4,5,6...]
        :return: all values of output layer nodes. e.g. [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.98]
        '''
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer -- outputs of hidden
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# test network
# if __name__ == '__main__':
#     n = neuralNetwork(3,3,3,0.3)
#     print(n.query([1.0, 0.5, -1.5]))