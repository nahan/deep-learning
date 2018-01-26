from demo01_perceptron import Perceptron

f = lambda x: x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    input_vecs = [[5, 8, 3, 6], [3, 3, 2, 2], [8, 8, 8, 8], [1.4, 2, 2, 1], [10.1, 8, 8, 8]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    lu = LinearUnit(4)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    print('work 3.4 year, monthly salary = %.2f' % linear_unit.predict([3.4, 3, 2, 3]))
    print('work 15 year, monthly salary = %.2f' % linear_unit.predict([15, 8, 3, 8]))
    print('work 1.5 year, monthly salary = %.2f' % linear_unit.predict([1.5, 2, 2, 1]))
    print('work 6.3 year, monthly salary = %.2f' % linear_unit.predict([6.3, 3, 2, 7]))
