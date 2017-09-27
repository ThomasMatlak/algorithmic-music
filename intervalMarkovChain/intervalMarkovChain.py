#!/usr/bin/python3

import random
from collections import defaultdict
import sys
import music21 as m21

# us = m21.environment.UserSettings()
# us['musicxmlPath'] = "c:/Program Files (x86)/EasyABC/easy_abc.exe"
# m21.environment.set('midiPath', "c:/Program Files (x86)/EasyABC/easy_abc.exe")


def normalize_score(score):
    """ Convert a score to C Major/a minor """
    orig_key = score.analyze('key')

    new_tonic = 'C'  # we assume the key is only Major or minor
    if orig_key.mode == 'minor':
        new_tonic = 'a'

    i = m21.interval.Interval(orig_key.tonic, m21.pitch.Pitch(new_tonic))

    return score.transpose(i)


def get_notes(stream):
    """ Return a list of the scale degrees of notes in a stream """

    notes = []

    for note in stream.getElementsByClass(m21.note.Note):
        notes.append(note)

    return notes


def main():
    my_score = m21.converter.parse(sys.argv[1])

    my_normalized_score = normalize_score(my_score)

    # create a Markov Chain
    # transition matrix
    transition_counts = defaultdict(lambda :defaultdict(int))

    prev_note = None
    interval = None

    for note in my_normalized_score[0].getElementsByClass(m21.note.Note):
        if prev_note is None:
            prev_note = note
            continue
        elif interval is None:
            interval = m21.interval.Interval(prev_note, note)
            prev_note = note
            continue
        else:
            current_interval = m21.interval.Interval(prev_note, note)
            transition_counts[interval.directedName][current_interval.directedName] += 1
            interval = current_interval
            prev_note = note

    generated_score = m21.stream.Score()
    generated_score.insert(0, m21.metadata.Metadata())
    generated_score.metadata.title = "Melody based on Bach's Little Fugue in g minor"
    generated_score.metadata.composer = "Markov Chain"
    generated_score.append(m21.tempo.MetronomeMark(number=random.randint(40, 168)))

    # seed the melody with the first two notes; melody will start on tonic in the 5th octave
    generated_notes = [m21.note.Note(m21.pitch.Pitch(9, octave=5), quarterLength=1)]
    interval = m21.interval.Interval(list(transition_counts.items())[random.randint(0, len(transition_counts))][0])
    interval.noteStart = m21.note.Note(m21.pitch.Pitch(9, octave=5), quarterLength=1)
    generated_notes.append(interval.noteEnd)

    count = 2

    # we want the melody to be a multiple of 4 beats that is >= 8 beats, ending on a pitch in the i chord
    while count < 16 or count % 4 != 0 or not (generated_notes[count - 1].pitch.pitchClass == 9 or
                                               generated_notes[count - 1].pitch.pitchClass == 0 or
                                               generated_notes[count - 1].pitch.pitchClass == 3):
        n = 0
        for p in transition_counts[interval.directedName]:
            n += transition_counts[interval.directedName][p]

        current_probabilities = {}
        for key in list(transition_counts[interval.directedName].keys()):
            current_probabilities[key] = float(transition_counts[interval.directedName][key] / float(n))

        rand_num = random.random()

        probability_sum = 0.0

        for index in current_probabilities:
            probability_sum += current_probabilities[index]

            if rand_num <= probability_sum:
                interval = m21.interval.Interval(index)
                interval.noteStart = generated_notes[count - 1]
                generated_notes.append(interval.noteEnd)
                break

        count += 1

    part = m21.stream.Part()
    part.offset = 0.0
    part.append(generated_notes)

    generated_score.append(part)

    generated_score.show('text')


if __name__ == '__main__':
    main()
