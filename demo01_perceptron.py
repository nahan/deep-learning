class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in xrange(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        zip_vec = zip(input_vec, self.weights)
        map_vec = map(lambda (x, w): x * w, zip_vec)
        sum = reduce(lambda a, b: a + b, map_vec)
        return self.activator(sum + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in xrange(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = map(lambda (x, w): rate * delta * x + w, zip(input_vec, self.weights))
        self.bias += rate * delta

def f(x):
    return 1 if x > 0 else 0

def get_and_training_dataset():
    input_vecs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]
    return input_vecs, labels

def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_and_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

def get_or_training_dataset():
    input_vecs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 1]
    return input_vecs, labels

def train_or_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_or_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('1 and 1 = %d' % and_perception.predict([1, 1]))

    or_perception = train_or_perceptron()
    print(or_perception)
    print('0 and 0 = %d' % or_perception.predict([0, 0]))
    print('0 and 1 = %d' % or_perception.predict([0, 1]))
    print('1 and 0 = %d' % or_perception.predict([1, 0]))
    print('1 and 1 = %d' % or_perception.predict([1, 1]))