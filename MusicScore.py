"""
MusicScore module

Provides a way to represent musical scores in Python, and operations to perform with the score.
Contains 
"""

# Thomas Matlak

class Note:
    """

    """

    def __init__(self, degree, octave, length):
        """

        """

        self.degree = degree
        self.octave = octave
        self.length = length

    def get_abc(self, key):
        """Converts the note to an abc representation and returns a string"""

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
