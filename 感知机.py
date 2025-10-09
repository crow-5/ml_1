import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        # 计算 w·x + b
        linear_output = np.dot(x, self.weights) + self.bias
        # 使用阶跃函数进行二分类
        y_pred = 1 if linear_output > 0 else 0
        return y_pred

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]
                y_pred = self.predict(x_i)
                if y_pred != y_i:
                    # 更新权重和偏置
                    self.weights += self.learning_rate * y_i * x_i
                    self.bias += self.learning_rate * y_i

    def evaluate(self, X, y):
        correct = 0
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]
            y_pred = self.predict(x_i)
            if y_pred == y_i:
                correct += 1
        accuracy = correct / len(X)
        return accuracy