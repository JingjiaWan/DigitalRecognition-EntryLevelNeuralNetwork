'''
Adjust parameters to view network performance
'''

import DigitalRecognition
import matplotlib.pyplot as plt
import numpy as np
import time

def plot(x, performance_y, time_y, parameter):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(parameter)
    ax1.set_ylabel('performance', color=color)
    ax1.plot(x, performance_y, color=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('time', color=color)
    ax2.plot(x, time_y, color=color)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    input_nodes = 784
    # hidden_nodes = 100
    output_nodes = 10
    # learning_rate = 0.3

    # test learning rate
    performance_y = []
    time_y = []
    # x = np.arange(0.1, 1, 0.1)
    # for lr in x:
    #     print("开始lr={}的执行".format(lr))
    #     start_time = time.time()
    #     performance = DigitalRecognition.DigitalRecognition(input_nodes, 100, output_nodes, lr, 1)
    #     end_time = time.time()
    #     performance_y.append(performance)
    #     time_y.append(end_time - start_time)
    # plot(x,performance_y,time_y,"learning rate")

    # test the number of hidden layer nodes
    # performance_y.clear()
    # time_y.clear()
    # x = range(50, 501, 50)
    # for h_node in x:
    #     print("开始h_node={}的执行".format(h_node))
    #     start_time = time.time()
    #     performance = DigitalRecognition.DigitalRecognition(input_nodes, h_node, output_nodes, 0.3, 1)
    #     end_time = time.time()
    #     performance_y.append(performance)
    #     time_y.append(end_time - start_time)
    # plot(x,performance_y,time_y,"h_node")

    # test epoch
    performance_y.clear()
    time_y.clear()
    x = range(1, 21, 5)
    for epoch in x:
        print("开始epoch={}的执行".format(epoch))
        start_time = time.time()
        performance = DigitalRecognition.DigitalRecognition(input_nodes, 200, output_nodes, 0.3, epoch)
        end_time = time.time()
        performance_y.append(performance)
        time_y.append(end_time - start_time)
    plot(x, performance_y, time_y, "epoch")