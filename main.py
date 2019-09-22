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

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(
		labels=Y, logits=output_layer))  # loss function

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


# MARK : - Training and Testing

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))  # returns list of Booleans
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# trains and prints loss and accuracy on mini batches of 100 iterations
for i in range(num_iterations):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	session.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob: droupout})

	# should not expect increasing accuracy here because values are per batch only
	if not i % 100:
		mini_batch_loss, mini_batch_accuracy = sess.run(
			[cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
		print(f'Iteration {i}\t|Loss = {mini_batch_loss}\t|Accuracy = {mini_batch_accuracy}')
print("\n")

# run on test set
test_accuracy = session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print(f'Accuracy on test set: {test_accuracy}')

