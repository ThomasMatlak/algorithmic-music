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
        """Converts the note to its abc representation and returns a str"""

        if self.degree is 0:
            return "z" * self.length # a degree of 0 represents a rest

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

class ScorePart:
    """

    """

    def __init__(self):
        self.instrument = ""

class MusicScore:
    """

    """

    def __init__(self, key, title, composer):
        """

        """

        self.concert_key = key
        self.title = title
        self.composer = composer

        self.parts = {}

    def get_abc(self):
        """Convert the score to abc notation and return a str."""

        return ""
