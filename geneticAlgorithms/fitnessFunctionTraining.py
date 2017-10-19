import tensorflow as tf
import os
import pickle
import glob
import music21 as m21
import pprint

learning_rate = 0.5
epochs = 10
batch_size = 20

# training data placeholders
# input x - 8 measures sampled at each sixteenth note = 128 inputs
x = tf.placeholder(tf.float32, [None, 128])
# output placeholder - fitness
y = tf.placeholder(tf.float32, [None, 1])

# hidden layer will have 500 nodes
W1 = tf.Variable(tf.random_normal([128, 500], stddev=0.03), name="W1")
b1 = tf.Variable(tf.random_normal([500]), name="b1")

W2 = tf.Variable(tf.random_normal([500, 1], stddev=0.03), name="W2")
b2 = tf.Variable(tf.random_normal([1]), name="b1")

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Get training data
training_file_names = glob.glob("*.mid.pickle")

real_measures = []

for f in training_file_names:
    score = pickle.load(open(f, "rb"))
    part = score[0].getElementsByClass(m21.note.Note)

    beats = 0.0
    count = 0
    measure = []

    while True:
        if beats >= 32: # 8 measures at a time
            break

        if count > len(part) - 1:
            measure += [0 for w in range(128 - len(measure))]
            break

        if beats + part[count].quarterLength > 32:
            beats = 32
        else:
            beats += part[count].quarterLength

        for x in range(int(part[count].quarterLength / 0.25)):
            measure.append(part[count].pitch.pitchClass)

        count += 1

    real_measures.append(measure)

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(real_measures) / batch_size)

    for epoch in range(epochs):
        avg_cost = 0
        # for i in range(total_batch):
            # batch_x = real_measures[i * batch_size:(i + 1) + batch_size]
        batch_x = real_measures[0:batch_size]
        batch_y = [1 for w in range(batch_size)]

        _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch

        print("Epoch:", (epoch + 1), "cost=", "{:.3f}".format(avg_cost))

    # print(sess.run(accuracy, feed_dict={x:}))
