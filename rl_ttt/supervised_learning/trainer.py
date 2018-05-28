"""
Module for NN trainer
"""
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


class Trainer(object):
    marker_mapping = {
        '': [1, 0, 0], 'X': [0, 1, 0], 'O': [0, 0, 1],
    }

    action_mapping = np.eye(9)

    def __init__(self):
        self.data = []

    def load(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        for line in tqdm(lines, desc='Loading nn data', position=0):
            state, coeff = line.split(';')
            coeff = float(coeff)
            state, action = state[1:-1].split('),')

            state = [s[1:-1] for s in state[1:].split(',')]
            state = [sp for s in state for sp in self.marker_mapping[s]]

            action = [int(x) for x in self.action_mapping[int(action)]]

            nn_input = tuple([*state, *action])
            nn_out = coeff

            self.data.append((nn_input, nn_out))

    def train(self):
        X, Y = data_to_XY(self.data)

        model = get_nn_model(n_inputs=len(X[0]), n_outputs=1)

        history = model.fit(X, Y, batch_size=5, epochs=100,
                            validation_split=0.25, verbose=1)

        model.save(f'data/nn_model.save')
        make_plots(history)


def make_plots(history):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    y_ulim = max(history.history['mean_squared_error'][0],
                 history.history['val_mean_squared_error'][0],
                 history.history['mean_absolute_error'][0],
                 history.history['val_mean_absolute_error'][0])

    ax1.set_ylim((0, y_ulim * 1.1))
    ax2.set_ylim((0, y_ulim * 1.1))

    make_plot(history, 'mean_squared_error', ax1)
    make_plot(history, 'mean_absolute_error', ax2)

    fig.tight_layout()
    fig.savefig(f'data/nn_plots.png')


def make_plot(history, metric_name, ax):
    ax.plot(history.history[metric_name], label='Train')
    ax.plot(history.history['val_' + metric_name], label='Test')

    ax.set_title('Model ' + metric_name)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Epoch')

    ax.legend(loc='upper left')


def data_to_XY(data):
    data = np.array(data)
    X, Y = data[:, 0], data[:, 1]

    X = np.array([list(x) for x in X])
    Y = np.array([list([y]) for y in Y])

    return X, Y


def get_nn_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(n_inputs, input_dim=n_inputs,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(n_inputs * 0.5),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='normal', activation='tanh'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    return model
