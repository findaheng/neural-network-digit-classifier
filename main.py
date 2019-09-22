import tensorflow as tensorflow
from tensorflow.examples.tutorials.mnist import input_data


# MARK : - Data Processing

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_training = mnist.train.num_examples  # 55,000
num_validation = mnist.validation.num_examples  #5,000
num_testing = mnist.test.num_examples  #10,000


# MARK : - Network Architecture

num_input = 784  # since 28x28 pixels
num_hidden1 = 512
num_hidden2 = 256
num_hidden3 = 128
num_output = 10  #since 0-9 digits


# MARK : - Hyperparameters

learning_rate = 1e-4  # small value to avoid overshooting
num_iterations = 1000
batch_size = 128  # number of training examples to use at each step
dropout = 0.5  # to help overfitting


# MARK : - Computational Graph

X = tf.placeholder("float", [None, num_input]) 
Y = tf.placeholder("float", [None, num_output])
keep_prob = tf.placeholder(tf.float32)  # to control the dropout rate
weights = {  # random values close to zero
	'w1': tf.Variable(tf.truncated_normal([num_input, num_hidden1], stddev=0.1)),
	'w2': tf.Variable(tf.truncated_normal([num_input, num_hidden2], stddev=0.1)),
	'w3': tf.Variable(tf.truncated_normal([num_input, num_hidden3], stddev=0.1)),
	'out': tf.Variable(tf.truncated_normal([num_input, num_output], stddev=0.1)),
}
biases = {  # small constant value to ensure tensors activate in initial stages
	'b1': tf.Variable(tf.constant(0.1, shape=[num_hidden1])),
	'b2': tf.Variable(tf.constant(0.1, shape=[num_hidden2])),
	'b3': tf.Variable(tf.constant(0.1, shape=[num_hidden3])),
	'out': tf.Variable(tf.constant(0.1, shape=[num_output])),

}

