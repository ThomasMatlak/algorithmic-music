#!/usr/bin/python3

import random
import sys
import music21 as m21

us = m21.environment.UserSettings()
us['musicxmlPath'] = "c:/Program Files (x86)/EasyABC/easy_abc.exe"
m21.environment.set('midiPath', "c:/Program Files (x86)/EasyABC/easy_abc.exe")

def normalize_score(score):
    ''' Convert a score to C Major/a minor '''
    orig_key = score.analyze('key')

    new_tonic = 'C' # we assume the key is only Major or minor
    if orig_key.mode == 'minor':
        new_tonic = 'a'

    i = m21.interval.Interval(orig_key.tonic, m21.pitch.Pitch(new_tonic))

    return score.transpose(i)

def get_pitch_classes(stream):
    ''' Return a list of the scale degrees of notes in a stream '''

    notes = []

    for note in stream.getElementsByClass(m21.note.Note):
        notes.append(note.pitch.pitchClass)

    return notes

def pretty_print_2d_list(my_list):
    for sub_list in my_list:
        print(sub_list)

def main():
    my_score = m21.converter.parse(sys.argv[1])

    my_normalized_score = normalize_score(my_score)
    my_pitch_classes = get_pitch_classes(my_normalized_score[0])

    ### create a Markov Chain
    # transition matrix
    transition_counts = [[0 for x in range(0, 12)] for y in range(0, 12)]

    current_pitch = my_pitch_classes[0]

    for i in range(0, len(my_pitch_classes) - 1):
        next_pitch = my_pitch_classes[i + 1]

        transition_counts[current_pitch][next_pitch] += 1
        current_pitch = next_pitch

    generated_pitch_classes = [9]
    count = 1

    # we want the melody to be a multiple of 4 beats that is >= 8 beats, ending on a pitch in the i chord
    while count < 16 or count % 4 != 0 or not (generated_pitch_classes[count - 1] == 9 or generated_pitch_classes[count - 1] == 0 or generated_pitch_classes[count - 1] == 3):
        prev = generated_pitch_classes[count - 1]

        current_transitions = transition_counts[prev]
        n = sum(current_transitions)
        current_probabilities = [c / float(n) for c in current_transitions]

        rand_num = random.random()

        probability_sum = 0.0

        for index, prob in enumerate(current_probabilities):
            probability_sum += prob

            if rand_num <= probability_sum:
                generated_pitch_classes.append(index)
                break

        count += 1

    generated_score = m21.stream.Score()
    generated_score.insert(0, m21.metadata.Metadata())
    generated_score.metadata.title = "Melody based on Bach's Little Fugue in g minor"
    generated_score.metadata.composer = "Markov Chain"

    part = m21.stream.Part()
    part.offset = 0.0

    for pitch_class in generated_pitch_classes:
        part.append(m21.note.Note(m21.pitch.Pitch(pitch_class), type='quarter'))

    generated_score.append(part)

    generated_score.show()

if __name__ == '__main__':
    main()
