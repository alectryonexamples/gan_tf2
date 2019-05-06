import numpy as np
import tensorflow as tf

class PointDataset(object):
    def __init__(self, data_type="quadratic"):
        if data_type == "quadratic":
            self.sample = self.generate_quadratic_points
        elif data_type == "circle":
            self.sample = self.generate_circle_points
        elif data_type == "sin":
            self.sample = self.generate_sin_points
        else:
            raise Exception("Not a valid data type")

    def generate_circle_points(self, n, radius=1.):
        thetas = np.random.sample(n) * 2*np.pi
        x = radius * np.cos(thetas)
        y = radius * np.sin(thetas)

        data = np.stack((x, y), axis=1)
        return data

    def generate_quadratic_points(self, n):
        x = np.random.sample(n) * 2 - 1
        y = np.power(x, 2)
        data = np.stack((x, y), axis=1)
        return data

    def generate_sin_points(self, n):
        x = np.random.sample(n) * 2*np.pi - np.pi
        freq = 1.5
        y = np.sin(freq*x)
        data = np.stack((x, y), axis=1)
        return data

    def dim(self):
        return 2


class MNISTDataset(object):
    def __init__(self, num=None, flatten=False):
        data = tf.keras.datasets.mnist.load_data()
        self.x_train = data[0][0]
        self.y_train = data[0][1]

        self.x_test = data[1][0]
        self.y_test = data[1][1]

        # turning into floats from 0 to 1
        self.x_train = np.array(self.x_train, dtype=np.float32) / 255
        self.x_test = np.array(self.x_test, dtype=np.float32) / 255

        self.numbered_x_train = []
        self.numbered_x_test = []
        for i in range(10):
            self.numbered_x_train.append(self.x_train[self.y_train == i, :, :])
            self.numbered_x_test.append(self.x_test[self.y_test == i, :, :])

        self.num = num
        self.flatten = flatten

    def sample(self, n):
        if self.num is None:
            rand_idx = np.random.randint(len(self.x_train), size=n)
            data = self.x_train[rand_idx, :, :]
        else:
            rand_idx = np.random.randint(len(self.numbered_x_train[self.num]), size=n)
            data = self.numbered_x_train[self.num][rand_idx, :, :]

        if self.flatten:
            return np.reshape(data, (n, -1))
        else:
            return np.expand_dims(data, -1)

    def dim(self):
        return 784


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = MNISTDataset()
    data = dataset.sample(2, num=2)
    print(data.shape)
    plt.imshow(data[0, :, :, 0])
    plt.show()

