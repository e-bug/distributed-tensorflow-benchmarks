from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ================================================================== #
#                             LOAD DATA                              #
# ================================================================== #
# Generate some data as y=3*x + noise
N_SAMPLES = 10
x_in = np.arange(N_SAMPLES)
y_in = 3*x_in + np.random.randn(N_SAMPLES)
data = list(zip(x_in, y_in))

# ================================================================== #
#                            BUILD GRAPH                             #
# ================================================================== #
simple_graph = tf.Graph()
with simple_graph.as_default():
  # Generate placeholders for input x and output y
  x = tf.placeholder(tf.float32, name='x')                                
  y = tf.placeholder(tf.float32, name='y')

  # Create weight and bias, initialized to 0
  w = tf.Variable(0.0, name='weight')
  b = tf.Variable(0.0, name='bias')

  # Build model to predict y
  y_predicted = x * w + b 

  # Use the square error as the loss function
  loss = tf.square(y - y_predicted, name='loss')

  # Use gradient descent to minimize loss
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train = optimizer.minimize(loss)

# ================================================================== #
#                           EXECUTE GRAPH                            #
# ================================================================== #
# Run training for N_EPOCHS epochs
N_EPOCHS = 5
with tf.Session(graph=simple_graph) as sess:
  # Initialize the necessary variables (w and b here) 
  sess.run(tf.global_variables_initializer())

  # Train the model
  for i in range(N_EPOCHS):
    total_loss = 0
    for x_,y_ in data:
      # Session runs train operation and fetches values of loss
      _, l_value = sess.run([train, loss], feed_dict={x: x_, y: y_}) 
      total_loss += l_value
    print('Epoch {0}: {1}'.format(i, total_loss/N_SAMPLES))

  # Output final values of w and b
  w_value, b_value = sess.run([w, b]) 

# ================================================================== #
#                             PLOT RESULTS                           #
# ================================================================== #
print(w_value, b_value)
plt.plot(x_in, y_in, 'bo', label='Real data')
plt.plot(x_in, x_in*w_value + b_value, 'orange', label='Predicted data')
plt.ylabel('y')
plt.xlabel('x')
plt.title('Linear Regression')
plt.legend()
plt.grid()
plt.show()
