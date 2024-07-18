import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist

dummy_images = np.random.rand(5, 28*28).astype(np.float32)
input_filename = '/content/input.txt'
with open(input_filename, 'w') as file:
    for image in dummy_images:
        file.write(' '.join(map(str, image)) + '\n')

input_filename
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 28*28).astype(np.float32) / 255.0

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))
operations = ['+', '-', '*', 'mean', 'max', 'min']

def create_random_tree(depth=3):
    if depth == 0 or random.random() > 0.5:
        return ('const', np.random.rand())
    else:
        op = random.choice(operations)
        if op in ['+', '-', '*']:
            return (op, create_random_tree(depth-1), create_random_tree(depth-1))
        else:
            return (op, create_random_tree(depth-1))

def tree_to_function(tree, X):
    if tree[0] == 'const':
        return tree[1]
    elif tree[0] == '+':
        return tree_to_function(tree[1], X) + tree_to_function(tree[2], X)
    elif tree[0] == '-':
        return tree_to_function(tree[1], X) - tree_to_function(tree[2], X)
    elif tree[0] == '*':
        return tree_to_function(tree[1], X) * tree_to_function(tree[2], X)
    elif tree[0] == 'mean':
        return np.mean(tree_to_function(tree[1], X))
    elif tree[0] == 'max':
        return np.max(tree_to_function(tree[1], X))
    elif tree[0] == 'min':
        return np.min(tree_to_function(tree[1], X))
