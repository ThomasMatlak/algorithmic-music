"""
Code to alter music21 streams

Includes functions for these musical transformations:
transpose
inverse
retrograde
inverse-retrograde
retrograde-inverse
crossover
"""

import music21 as m21


def transpose(stream, interval=0):
    """ Transpose the Stream by the specified interval """
    return stream.transpose(interval)


def inverse(stream):
    """ Reverse the intervals of the Stream """

    notes = stream.flat.getElementsByClass(m21.note.Note)
    rhythms = [note.quarterLength for note in notes]

    orig_intervals = []
    for i in range(len(notes) - 1):
        orig_intervals.append(m21.interval.notesToInterval(notes[i], notes[i + 1]))

    reverse_intervals = [interval.reverse() for interval in orig_intervals]

    return_stream = m21.stream.Stream()
    return_stream.append(notes[0])
    return_stream[0].quarterLength = rhythms[0]

    for i, interval in enumerate(reverse_intervals):
        new_note = interval.transposeNote(return_stream[i])
        new_note.quarterLength = rhythms[i + 1]
        return_stream.append(new_note)

    return return_stream


def retrograde(stream, reverse_notes = True, reverse_rhythms=True):
    """ Reverse the notes in the Stream """

    pitches = [note.pitch for note in stream.flat.notes]
    rhythms = [note.quarterLength for note in stream.flat.notes]

    if reverse_notes:
        pitches = list(reversed(pitches))
    if reverse_rhythms:
        rhythms = list(reversed(rhythms))

    return_stream = m21.stream.Stream()

    for i in range(len(pitches)):
        note = m21.note.Note(pitches[i])
        note.quarterLength = rhythms[i]
        return_stream.append(note)

    return return_stream


def retrograde_inverse(stream):
    return retrograde(inverse(stream))


def inverse_retrograde(stream):
    return inverse(retrograde(stream))


def crossover(parent1, parent2, crossover_points=[]):
    """ Perform the GA operation crossover on two Streams """
    crossover_points.sort()
    parent1 = parent1.flat.notes
    parent2 = parent2.flat.notes

    child1 = []
    child2 = []

    prev_crossover_point = 0

    for i, crossover_point in enumerate(crossover_points):
        if i % 2 is 0:
            child1 += parent1[prev_crossover_point:crossover_point]
            child2 += parent2[prev_crossover_point:crossover_point]
        else:
            child1 += parent2[prev_crossover_point:crossover_point]
            child2 += parent1[prev_crossover_point:crossover_point]

        prev_crossover_point = crossover_point

    if len(crossover_points) % 2 is 0:
        child1 += parent1[prev_crossover_point:]
        child2 += parent2[prev_crossover_point:]
    else:
        child1 += parent2[prev_crossover_point:]
        child2 += parent1[prev_crossover_point:]

    return m21.stream.Stream(child1), m21.stream.Stream(child2)
