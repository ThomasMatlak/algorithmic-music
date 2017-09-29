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


def create_transition_matrices(streams):
    """ create Markov Chains for intervals and rhythms """
    interval_transitions = defaultdict(lambda: defaultdict(int))
    rhythm_transitions = defaultdict(lambda: defaultdict(int))

    for stream in streams:
        prev_note = None
        interval = None

        for note in stream:
            if prev_note is None:
                prev_note = note
                continue
            elif interval is None:
                interval = m21.interval.Interval(prev_note, note)
                prev_note = note
                continue
            else:
                current_interval = m21.interval.Interval(prev_note, note)
                interval_transitions[interval.directedName][current_interval.directedName] += 1
                interval = current_interval
                rhythm_transitions[prev_note.quarterLength][note.quarterLength] += 1
                prev_note = note

    return interval_transitions, rhythm_transitions


def main():
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

    transition_matrices = create_transition_matrices([s[0].getElementsByClass(m21.note.Note) for s in normalized_scores])
    interval_transitions = transition_matrices[0]
    rhythm_transitions = transition_matrices[1]

    # Set up the score
    generated_score = m21.stream.Score()
    generated_score.insert(0, m21.metadata.Metadata())
    generated_score.metadata.composer = "Markov Chain"
    generated_score.append(m21.tempo.MetronomeMark(number=random.randint(40, 168)))

    # seed the melody with the first two notes; melody will start on tonic in the 5th octave
    generated_notes = [m21.note.Note(m21.pitch.Pitch(0, octave=4), quarterLength=1)]
    interval = m21.interval.Interval(list(interval_transitions.items())[random.randrange(0, len(interval_transitions))][0])
    generated_notes.append(generated_notes[0].transpose(interval))
    generated_notes[1].quarterLength = list(rhythm_transitions.items())[random.randrange(0, len(rhythm_transitions))][0]

    count = 2
    beats = 2.0

    # toggle between these while statements to get a major or minor end note
    # we want the melody to be a multiple of 4 beats that is >= 8 beats, ending on a pitch in the i chord
    # while beats < 32 or beats % 4 != 0 or not (generated_notes[count - 1].name == 'A' or # minor
    #                                            generated_notes[count - 1].name == 'C' or
    #                                            generated_notes[count - 1].name == 'E'):
    while beats < 32 or beats % 4 != 0 or not (generated_notes[count - 1].name == 'C' or # Major
                                               generated_notes[count - 1].name == 'E' or
                                               generated_notes[count - 1].name == 'G'):
        transition_sum = 0.0
        for p in interval_transitions[interval.directedName]:
            transition_sum += interval_transitions[interval.directedName][p]

        rhythm_sum = 0.0
        for p in rhythm_transitions[generated_notes[count - 1].quarterLength]:
            rhythm_sum += rhythm_transitions[generated_notes[count - 1].quarterLength][p]

        curr_interval_probabilities = {}
        for key in list(interval_transitions[interval.directedName].keys()):
            curr_interval_probabilities[key] = interval_transitions[interval.directedName][key] / transition_sum

        curr_rhythm_probabilities = {}
        for key in list(rhythm_transitions[generated_notes[count - 1].quarterLength].keys()):
            curr_rhythm_probabilities[key] = rhythm_transitions[generated_notes[count - 1].quarterLength][key] / rhythm_sum

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
    main()
