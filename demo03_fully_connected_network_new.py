import numpy as np


# input_size = 5
# output_size = 3
# w = np.random.uniform(-0.1, 0.1, (output_size, input_size))
# b = np.zeros((output_size, 1))
# output = np.zeros((output_size, 1))
#
# print(w)
# print(b)
# print(output)
#
# print(np.shape(w))
# print(np.shape(b))
# print(np.shape(output))


class FullyConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, self.input) + self.b)

    def backward(self, delta_array):
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


class SigmoidActivator(object):
    def forward(self, weighted_input):
        np.seterr(over='ignore')
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in xrange(len(layers) - 1):
            self.layers.append(FullyConnectedLayer(layers[i], layers[i + 1], SigmoidActivator()))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        for i in xrange(epoch):
            for d in xrange(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)


def transpose(args):
    return map(
        lambda arg: map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg)
        , args
    )
