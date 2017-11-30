import music21 as m21
from collections import defaultdict

class MarkovChain(object):
    """  """

    def __init__(self, order=1):
        """  """
        self.order = order
        self.transition_matrix = self.generate_nested_defaultdict(self.order + 1)

    def generate_nested_defaultdict(self, depth):
        """ Create a nested collections.defaultdict object with depth `depth` """
        if depth == 1:
            return defaultdict(int)
        else:
            return defaultdict(lambda: self.generate_nested_defaultdict(depth - 1))


    def arbitrary_depth_dict_get(self, subscripts, default=None, nested_dict=-1):
        """ Access nested dict elements at arbitrary depths

            https://stackoverflow.com/questions/28225552/is-there-a-recursive-version-of-pythons-dict-get-built-in
        """
        if nested_dict == -1:
            nested_dict = self.transition_matrix

        if not subscripts:
            return nested_dict

        key = subscripts[0]

        if isinstance(nested_dict, int):
            return nested_dict

        return self.arbitrary_depth_dict_get(subscripts[1:], default, nested_dict.get(key, default))


    def get_transitions_from_state(self, prev_intervals):
        """ A wrapper function for arbitrary_depth_dict_get """
        return self.arbitrary_depth_dict_get(prev_intervals, {})


    def arbitrary_depth_dict_set(self, subscripts, _dict={}, val=None):
        """ Set nested dict elements at arbitrary depths

            https://stackoverflow.com/questions/33663332/python-adding-updating-dict-element-of-any-depth
        """

        if not subscripts:
            return _dict
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

    def create_transition_matrix(self, streams, chainType):
        """  """
        for stream in streams:
            prev_notes = []

            for n in range(len(stream)):
                note = stream[n]

                if len(prev_notes) < (self.order + 1):
                    prev_notes.append(note)
                    continue
                else:
                    if chainType == "interval" or chainType == "i":
                        intervals = []
                        for i in range(self.order):
                            intervals.append(m21.interval.Interval(prev_notes[i], prev_notes[i + 1]))

                        intervals.append(m21.interval.Interval(prev_notes[-1], note))

                        self.transition_matrix = self.arbitrary_depth_dict_set(
                            [interval.directedName for interval in intervals[0:self.order + 1]],
                            self.transition_matrix,
                            self.arbitrary_depth_dict_get([interval.directedName for interval in intervals[0:self.order + 1]], default=0) + 1
                        )
                    elif chainType == "rhythm" or chainType == "r":
                        self.transition_matrix = self.arbitrary_depth_dict_set(
                            [prev_note.quarterLength for prev_note in prev_notes],
                            self.transition_matrix,
                            self.arbitrary_depth_dict_get([prev_note.quarterLength for prev_note in prev_notes], default=0) + 1
                        )

                    for i in range(self.order):
                        prev_notes[i] = prev_notes[i + 1]

                    prev_notes[self.order] = note
