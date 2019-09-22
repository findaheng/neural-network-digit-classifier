import tensorflow as tensorflow
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_training = mnist.train.num_examples  # 55,000
num_validation = mnist.validation.num_examples  #5,000
num_testing = mnist.test.num_examples  #10,000

num_input = 784  # since 28x28 pixels
num_hidden1 = 512
num_hidden2 = 256
num_hidden3 = 128
num_output = 10  #since 0-9 digits

