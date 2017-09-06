"""
MusicScore module

Provides a way to represent musical scores in Python, and operations to perform with the score.
Contains the classes Note, Measure, ScorePart, and MusicScore
"""

# Thomas Matlak

MAJOR_SCALES = {
    "Cb": ["_c", "_d", "_e", "_f", "_g", "_a", "_b"],
    "Gb": ["_g", "_a", "_b", "_c", "_d", "_e", "f"],
    "Db": ["_d", "_e", "f", "_g", "_a", "_b", "c"],
    "Ab": ["_a", "_b", "c", "_d", "_e", "f", "g"],
    "Eb": ["_e", "f", "g", "_a", "_b", "c", "d"],
    "Bb": ["_b", "c", "d", "_e", "f", "g", "a"],
    "F": ["f", "g", "a", "_b", "c", "d", "e"],
    "C": ["c", "d", "e", "f", "g", "a", "b"],
    "G": ["g", "a", "b", "c", "d", "e", "^f"],
    "D": ["d", "e", "^f", "g", "a", "b", "^c"],
    "A": ["a", "b", "^c", "d", "e", "^f", "^g"],
    "E": ["e", "^f", "^g", "a", "b", "^c", "^d"],
    "B": ["b", "^c", "^d", "e", "^f", "^g", "^a"],
    "F#": ["^f", "^g", "^a", "b", "^c", "^d", "^e"],
    "C#": ["^c", "^d", "^e", "^f", "^g", "^a", "^b"]
}

class Note:
    """

    """

    def __init__(self, degree, octave, length):
        """

        """

        self.degree = degree
        self.octave = octave
        self.length = length # a multiple of the unit note length, defined by the MusicScore class

    def get_abc(self, key):
        """Convert the note to abc notation and return a str"""

        if self.degree is 0:
            return "z" + str(self.length) # a degree of 0 represents a rest

        if self.octave > 5: # two octaves above the middle octave
            octave_modifier = "'" * (self.octave - 5)
        elif self.octave < 4: # below the middle octave
            octave_modifier = "," * (4 - self.octave)
        elif self.octave == 4: # the middle octave
            octave_modifier = ""

        abc_note = MAJOR_SCALES[key][self.degree - 1] # scale degrees are not 0 indexed

        if self.octave <= 4:
            abc_note = abc_note.upper()

        return abc_note + octave_modifier + str(self.length)

class Measure:
    """

    """

    def __init__(self):
        self.contents = []

    def add_note(self, degree, octave, length):
        """Append a note to the end of the measure"""

        self.contents.append(Note(degree, octave, length))

    def get_abc(self, key):
        """Convert the entire measure to abc notation and return a str"""

        abc_measure = ""

        for note in self.contents:
            abc_measure += note.get_abc(key) + " "

        return abc_measure[:-2]

class ScorePart:
    """

    """

    def __init__(self, part_id, name, short_name, instrument, clef):
        """ """
        self.part_id = part_id
        self.name = name
        self.short_name = short_name
        self.instrument = instrument
        self.clef = clef
        self.measures = []

    def add_measure(self, measure=None):
        """Append a measure to the part"""

        if not measure is None:
            self.measures.append(measure)
        else:
            self.measures.append(Measure())

    def get_abc(self, key):
        """Convert the part to abc notation and return a str"""

        part_abc = "V: " + self.part_id + \
            " name=" + self.name + \
            " subname=" + self.short_name + \
            " clef=" + self.clef + "\n"

        for measure in self.measures:
            part_abc += " " + measure.get_abc(key) + " |"

        part_abc += "]"

        return part_abc

class MusicScore:
    """

    """

    def __init__(self, key, title, composer, meter, unit_note_length, tempo, reference_number=1):
        """

        """

        self.concert_key = key
        self.title = title
        self.composer = composer
        self.meter = meter
        self.unit_note_length = unit_note_length
        self.tempo = tempo
        self.reference_number = reference_number

        self.parts = {}

    def add_part(self, part=None):
        """Add a voice to the score"""

        if not part is None:
            self.parts[part.name] = part

    def get_abc(self):
        """Convert the score to abc notation and return a str."""

        score_abc = "%abc-2.1\n"
        score_abc += "X: " + str(self.reference_number) + "\n"
        score_abc += "T: " + self.title + "\n"
        score_abc += "C: " + self.composer + "\n"
        score_abc += "M: " + self.meter + "\n"
        score_abc += "L: " + self.unit_note_length + "\n"
        score_abc += "Q: " + self.tempo + "\n"
        score_abc += "K: " + self.concert_key + "\n"

        score_abc += "% end of header\n"

        for part_name, part in self.parts.items():
            score_abc += part.get_abc(self.concert_key) + "\n\n"

        return score_abc
