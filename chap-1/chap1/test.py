import chap1.mnist_loader
import chap1.network

training_data, validation_data, test_data = chap1.mnist_loader.load_data_wrapper()
net = chap1.network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
