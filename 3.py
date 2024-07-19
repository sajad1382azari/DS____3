import numpy as np 
import random 
from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras.datasets import mnist 

dummy_images = np.random.rand(5, 28*28).astype(np.float32) 
nput_filename = '/input.txt' 
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
