import music21 as m21

def transpose(stream, interval=0):
    """ transpose the stream by the specified interval """
    return stream.transpose(interval)


def inverse(stream):
    """ reverse the intervals of the stream """

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


def retrograde(stream, reverse_notes = True, reverse_rhythms = True):
    """ reverse the notes in the stream """

    pitches = [note.pitch for note in stream.flat.getElementsByClass(m21.note.Note)]
    rhythms = [note.quarterLength for note in stream.flat.getElementsByClass(m21.note.Note)]

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
