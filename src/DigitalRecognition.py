'''
DigitalRecognition Object
'''
import numpy
import NeuralNetwork

def loadData(path):
    with open(path,'r') as f:
        data_list = f.readlines()
    return data_list

def DigitalRecognition(input_nodes, hidden_nodes, output_nodes, learning_rate, epoch):

    # input_nodes = 784
    # hidden_nodes = 100
    # output_nodes = 10
    # learning_rate = 0.3
    # epoch = 1
    DigitalRec = NeuralNetwork.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

    # train network
    for e in range(epoch):
        inputs_list = loadData("../dataset/mnist_train.csv")
        for record in inputs_list:
            # split the record by the ',' commas : str -> list of str
            inputs = record.split(',')
            # scale inputs in [0.01, 1]
            scaled_inputs = (numpy.asfarray(inputs[1:]) / 255.0 * 0.99) + 0.01
            # target outputs : all 0.01, except the desired label which is 0.99
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(inputs[0])] = 0.99
            DigitalRec.train(scaled_inputs, targets)

    # test network
    tests_list = loadData("../dataset/mnist_test.csv")
    # scordcard for how well the network perform, initially empty
    scordcard = []
    for record in tests_list:
        inputs = record.split(',')
        scaled_inputs = (numpy.asfarray(inputs[1:]) / 255.0 * 0.99) + 0.01
        sample_output = int(inputs[0])
        outputs = DigitalRec.query(scaled_inputs)
        # the index of the highest value corresponds to the network_output
        network_output = numpy.argmax(outputs)
        if(sample_output == network_output):
            scordcard.append(1)
        else:
            scordcard.append(0)

    # calculate the performance score, the fraction of correct answers
    # print("performance = ", sum(scordcard) / len(scordcard))
    return sum(scordcard) / len(scordcard)