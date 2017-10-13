#!/usr/bin/python3

import random
from collections import defaultdict
import sys
import os
import pickle
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


def generate_nested_defaultdict(depth):
    """ Create a nested collections.defaultdict object with depth `depth` """
    if depth == 1:
        return defaultdict(int)
    else:
        return defaultdict(lambda: generate_nested_defaultdict(depth - 1))


def arbitrary_depth_get(d, subscripts, default=None):
    """ Access nested dict elements at arbitrary depths

        https://stackoverflow.com/questions/28225552/is-there-a-recursive-version-of-pythons-dict-get-built-in
    """
    if not subscripts:
        return d
    key = subscripts[0]
    if isinstance(d, int):
        return d
    return arbitrary_depth_get(d.get(key, default), subscripts[1:], default=default)


def arbitrary_depth_set(subscripts, _dict={}, val=None):
    """ Set nested dict elements at arbitrary depths

        https://stackoverflow.com/questions/33663332/python-adding-updating-dict-element-of-any-depth
    """

    if not subscripts:
        return _dict
    # subscripts = [s.strip() for s in subscripts]
    for sub in subscripts[:-1]:
        if '_x' not in locals():
            if sub not in _dict:
                _dict[sub] = {}
            _x = _dict.get(sub)
        else:
            if sub not in _x:
                _x[sub] = {}
            _x = _x.get(sub)
    _x[subscripts[-1]] = val
    return _dict


def create_interval_transition_matrix(streams, order):
    """ Create the interval transition matrix for an `order`th order Markov Chain """
    interval_transitions = generate_nested_defaultdict(order + 1)

    for stream in streams:
        prev_notes = []

        for n in range(len(stream)):
            note = stream[n]

            if len(prev_notes) < (order + 1):
                prev_notes.append(note)
                continue
            else:
                intervals = []
                for i in range(order):
                    intervals.append(m21.interval.Interval(prev_notes[i], prev_notes[i + 1]))

                intervals.append(m21.interval.Interval(prev_notes[-1], note))

                interval_transitions = arbitrary_depth_set(
                    [interval.directedName for interval in intervals[0:order + 1]],
                    interval_transitions,
                    arbitrary_depth_get(interval_transitions, [interval.directedName for interval in intervals[0:order + 1]], default=0) + 1
                )

                for i in range(order):
                    prev_notes[i] = prev_notes[i + 1]

                prev_notes[order] = note

    return interval_transitions


def create_rhythm_transition_matrix(streams, order):
    """ Create the rhythmic transition matrix for an `order`th order Markov Chain """
    rhythm_transitions = generate_nested_defaultdict(order + 1)

    for stream in streams:
        prev_notes = []

        for n in range(len(stream)):
            note = stream[n]

            if len(prev_notes) < (order + 1):
                prev_notes.append(note)
                continue
            else:
                rhythm_transitions = arbitrary_depth_set(
                    [prev_note.quarterLength for prev_note in prev_notes],
                    rhythm_transitions,
                    arbitrary_depth_get(rhythm_transitions, [prev_note.quarterLength for prev_note in prev_notes], default=0) + 1
                )
                # rhythm_transitions[prev_notes[1].quarterLength][note.quarterLength] += 1

                for i in range(order):
                    prev_notes[i] = prev_notes[i + 1]

                prev_notes[order] = note

    return rhythm_transitions


def main(interval_order, rhythm_order):
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

    interval_transitions = create_interval_transition_matrix(note_streams, interval_order)
    rhythm_transitions = create_rhythm_transition_matrix(note_streams, rhythm_order)

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

    # toggle between these while statements to get a major or minor end note
    # we want the melody to be a multiple of 4 beats that is >= 8 beats, ending on a pitch in the i chord
    # while beats < 32 or beats % 4 != 0 or not (generated_notes[count - 1].name == 'A' or # minor
    #                                            generated_notes[count - 1].name == 'C' or
    #                                            generated_notes[count - 1].name == 'E'):
    while beats < 32 or beats % 4 != 0 or not (generated_notes[count - 1].name == 'C' or # Major
                                               generated_notes[count - 1].name == 'E' or
                                               generated_notes[count - 1].name == 'G'):
        # interval
        prev_interval_names = []
        for i in range(interval_order):
            prev_interval_names.append(m21.interval.Interval(generated_notes[count - (interval_order + 1) + i], generated_notes[count - interval_order + i]).directedName)

        prev_note_lengths = []
        for r in range(rhythm_order):
            prev_note_lengths.append(generated_notes[count - interval_order + i].quarterLength)


        interval_subset = arbitrary_depth_get(interval_transitions, prev_interval_names, default={})
        rhythm_subset = arbitrary_depth_get(rhythm_transitions, prev_note_lengths, default={})


        interval_sum = 0.0
        for p in interval_subset:
            interval_sum += interval_subset[p]

        rhythm_sum = 0.0
        for p in rhythm_subset:
            rhythm_sum += rhythm_subset[p]


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
