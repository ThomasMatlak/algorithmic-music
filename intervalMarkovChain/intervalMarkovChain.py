#!/usr/bin/python3

import random
from collections import defaultdict
import sys
import os
import pickle
from markovChain import MarkovChain
import music21 as m21

us = m21.environment.UserSettings()
us['musicxmlPath'] = "c:/Program Files (x86)/EasyABC/easy_abc.exe"
m21.environment.set('midiPath', "c:/Program Files (x86)/EasyABC/easy_abc.exe")


def normalize_score(score):
    """ Convert a score to C Major/a minor """
    orig_key = score.analyze('key')

    new_tonic = 'C'  # we assume the key is only Major or minor
    if orig_key.mode == 'minor':
        new_tonic = 'a'

    i = m21.interval.Interval(orig_key.tonic, m21.pitch.Pitch(new_tonic))

    return score.transpose(i)


def main(interval_order, rhythm_order):
    min_beats = 32
    beats_per_measure = 4
    major = True

    score_titles = sys.argv[1:]

    normalized_scores = []

    if not os.path.isdir("scoreCache"):
        os.mkdir("scoreCache")

    for score_title in score_titles:
        if os.path.exists("scoreCache/" + score_title + ".pickle"):
            normalized_scores.append(pickle.load(open("scoreCache/" + score_title + ".pickle", "rb")))
        else:
            my_score = m21.converter.parse(score_title)
            normalized_scores.append(normalize_score(my_score))
            pickle.dump(normalized_scores[-1], open("scoreCache/" + score_title + ".pickle", "wb"))

    note_streams = [s[0].getElementsByClass(m21.note.Note) for s in normalized_scores]

    interval_markov_chain = MarkovChain(interval_order)
    interval_markov_chain.create_transition_matrix(note_streams, "i")

    rhythm_markov_chain = MarkovChain(rhythm_order)
    rhythm_markov_chain.create_transition_matrix(note_streams, "r")

    # Set up the score
    generated_score = m21.stream.Score()
    generated_score.insert(0, m21.metadata.Metadata())
    generated_score.metadata.composer = "Markov Chain"
    generated_score.append(m21.tempo.MetronomeMark(number=random.randint(40, 168)))

    # TODO Allow user to provide their own seed input
    # seed the melody with the first `interval_order` + 1 notes;
    # first notes
    generated_notes = [
        m21.note.Note(m21.pitch.Pitch(7, octave=3), quarterLength=0.25),
        m21.note.Note(m21.pitch.Pitch(2, octave=4), quarterLength=0.25),
        m21.note.Note(m21.pitch.Pitch(11, octave=4), quarterLength=0.25),
        m21.note.Note(m21.pitch.Pitch(9, octave=4), quarterLength=0.25)
    ]

    count = len(generated_notes)
    beats = sum([g.quarterLength for g in generated_notes])

    if major:
        ending_pitches = ('C', 'E', 'G')
    else:
        ending_pitches = ('A', 'C', 'E')

    while beats < min_beats or beats % beats_per_measure != 0 or not generated_notes[count - 1].name in ending_pitches:
        # interval
        prev_interval_names = []
        for i in range(interval_order):
            prev_interval_names.append(m21.interval.Interval(generated_notes[count - (interval_order + 1) + i], generated_notes[count - interval_order + i]).directedName)

        prev_note_lengths = []
        for r in range(rhythm_order):
            prev_note_lengths.append(generated_notes[count - interval_order + i].quarterLength)

        interval_subset = interval_markov_chain.arbitrary_depth_get(prev_interval_names)
        rhythm_subset = rhythm_markov_chain.arbitrary_depth_get(prev_note_lengths)

        interval_sum = 0.0
        for p in interval_subset:
            interval_sum += interval_subset[p]

        rhythm_sum = 0.0
        for p in rhythm_subset:
            rhythm_sum += p

        curr_interval_probabilities = {}
        for key in list(interval_subset.keys()):
            curr_interval_probabilities[key] = interval_subset[key] / interval_sum

        curr_rhythm_probabilities = {}
        for key in list(rhythm_subset.keys()):
            curr_rhythm_probabilities[key] = rhythm_subset[key] / rhythm_sum

        interval_rand_num = random.random()
        rhythm_rand_num = random.random()

        interval_probability_sum = 0.0
        rhythm_probability_sum = 0.0

        my_note = None

        for index in curr_interval_probabilities:
            interval_probability_sum += curr_interval_probabilities[index]

            if interval_rand_num <= interval_probability_sum:
                interval = m21.interval.Interval(index)
                my_note = generated_notes[count - 1].transpose(interval)
                break

        if my_note is None:
            print("Encountered a sequence of intervals not seen during training, exiting.")
            break

        for index in curr_rhythm_probabilities:
            rhythm_probability_sum += curr_rhythm_probabilities[index]

            if rhythm_rand_num <= rhythm_probability_sum:
                my_note.quarterLength = index
                break

        my_note.pitch = my_note.pitch.getEnharmonic().getEnharmonic()  # this prevents more that double sharps and flats
        generated_notes.append(my_note)

        beats += my_note.quarterLength
        count += 1

        # program has been running too long, end it
        if beats > 400:
            break

    part = m21.stream.Part()
    part.offset = 0.0
    part.append(generated_notes)

    generated_score.append(part)

    generated_score.show()


if __name__ == '__main__':
    main(3, 3)
