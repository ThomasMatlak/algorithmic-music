#!/usr/bin/python3

import tensorflow as tf
import os
import pickle
import glob
import numpy as np
import random
import music21 as m21

def normalize_score(score):
    """ Convert a score to C Major/a minor """
    orig_key = score.analyze('key')

    new_tonic = 'C'  # we assume the key is only Major or minor
    if orig_key.mode == 'minor':
        new_tonic = 'a'

    i = m21.interval.Interval(orig_key.tonic, m21.pitch.Pitch(new_tonic))

    return score.transpose(i)


def convert_part(part):
    # convert the part entirely to sixteenth notes
    converted_part = []
    for note in part:
        if note.quarterLength / 0.25 - int(note.quarterLength / 0.25) != 0:
            continue # TODO be able to handle triplets

        note_len = note.quarterLength
        note.quarterLength = 0.25

        converted_part += [note.pitch.pitchClass for i in range(int(note_len / 0.25))]

    return converted_part


def main():
    # 8 measures sampled at each sixteenth note = 128 inputs
    input_size = 128
    num_beats = input_size / 4
    num_measures = input_size / 16
    hidden_layer_size = 500

    proportion_of_training_vs_testing_data = 0.7

    learning_rate = 0.5
    epochs = 10
    batch_size = 21

    # training data placeholders
    x = tf.placeholder(tf.float32, [None, input_size])
    # output placeholder, indicates fitness
    y = tf.placeholder(tf.float32, [None, 2])

    # hidden layer will have 500 nodes
    W1 = tf.Variable(tf.random_normal([input_size, hidden_layer_size], stddev=0.03), name="W1")
    b1 = tf.Variable(tf.random_normal([hidden_layer_size]), name="b1")

    W2 = tf.Variable(tf.random_normal([hidden_layer_size, 2], stddev=0.03), name="W2")
    b2 = tf.Variable(tf.random_normal([2]), name="b1")

    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    init_op = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Get training data, give label 1
    training_file_names = glob.glob("scoreCache\*.mid.pickle")

    training_data = []
    training_labels = []

    # Parts in the score cache are assumed to already be normalized
    for f in training_file_names:
        score = pickle.load(open(f, "rb"))
        part = score[0].getElementsByClass(m21.note.Note)

        # convert the part entirely to sixteenth notes
        converted_part = convert_part(part)

        measures_in_piece = len(converted_part) // 16

        for offset in range(measures_in_piece - 7):
            sample = []

            for i in range(input_size):
                sample.append(converted_part[(offset * 16) + i])

            training_data.append(sample)
            training_labels.append([1, 0])

    # Generate an amount of junk data equal to the amount of good data
    for i in range(len(training_data)):
        training_data.append([random.randrange(0,11) for j in range(input_size)])
        training_labels.append([0, 1])

    # mix the good and bad data
    c = list(zip(training_data, training_labels))
    random.shuffle(c)
    training_data, training_labels = zip(*c)

    with tf.Session() as sess:
        sess.run(init_op)
        total_batch = int(len(training_data) * proportion_of_training_vs_testing_data)

        train = training_data[0:total_batch]
        train_label = training_labels[0:total_batch]

        test = training_data[total_batch:]
        test_label = training_labels[total_batch:]

        for epoch in range(epochs):
            avg_cost = 0

            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: train, y: train_label})
            avg_cost += c / total_batch

        print(sess.run(accuracy, feed_dict={x: test, y: test_label}))

        # save the model


        # try the NN with a generated piece
        # score = m21.converter.parse('arbitraryOrderTest.mid')
        # score = m21.converter.parse('partita_for_unaccompanied_flute_bwv-1013_2_(c)grossman.mid')
        score = m21.converter.parse('../corpus/suite_for_unaccompanied_cello_bwv-1007_1_(c)grossman.mid')
        score = normalize_score(score)
        score_notes = score[0].getElementsByClass(m21.note.Note)

        # convert the part entirely to sixteenth notes
        converted_part = convert_part(score_notes)

        prediction = tf.argmax(y_, 1)
        print(prediction.eval(feed_dict={x: [converted_part[:128]]}))


if __name__ == "__main__":
    main()
