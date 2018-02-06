#!/usr/bin/python3

import sys
sys.path.append("../intervalMarkovChain")
import intervalMarkovChain as markov
import lstm
import music21 as m21
import glob
import random
import mutations
import copy
import os
import time
import json
from threading import Thread
from queue import Queue, Empty, deque


POPULATION_SIZE = 25
MAX_GENERATIONS = 50
FITNESS_THRESHOLD = 5  # fitness is measured as the difference between the LSTM NN's scores of good and bad
SAVE_DATA = True
START_WITH_MARKOV = False


class MutationThread(Thread):
    def __init__(self, thread_id, population, new_population):
        Thread.__init__(self)
        self.id = thread_id
        self.population = population
        self.new_population = new_population

    def run(self):
        while True:
            try:
                melody = self.population.get(timeout=0.5)

                for i in range(0, 13):
                    if random.random() < 0.9:
                        self.new_population.put(mutations.transpose(melody, i))

                if random.random() < 0.7:
                    self.new_population.put(mutations.inverse(melody))
                if random.random() < 0.5:
                    self.new_population.put(mutations.inverse_retrograde(melody))
                if random.random() < 0.5:
                    self.new_population.put(mutations.retrograde_inverse(melody))
                if random.random() < 0.5:
                    self.new_population.put(mutations.retrograde(melody, True, True))
                if random.random() < 0.5:
                    self.new_population.put(mutations.retrograde(melody, True, False))
                if random.random() < 0.5:
                    self.new_population.put(mutations.retrograde(melody, False, True))
            except Empty:
                break


def compact_notes(stream):
    """ group repeated sixteenth notes into one longer note """

    new_stream = m21.stream.Stream()

    current_note = stream[0]
    current_note_repeats = 1

    for note in stream[1:]:
        if note == current_note:
            current_note_repeats += 1
        else:
            new_stream.append(m21.note.Note(current_note.pitch, quarterLength=current_note_repeats * 0.25))
            current_note = note
            current_note_repeats = 1

    # be sure to include the last note
    new_stream.append(m21.note.Note(current_note.pitch, quarterLength=current_note_repeats * 0.25))

    return new_stream


def split_notes_into_sixteenth_notes(stream):
    """ splits notes into a number of sixteenth notes such that the melodic rhythm stays the same """

    new_stream = m21.stream.Stream()

    for note in stream:
        if note.quarterLength / 0.25 - int(note.quarterLength / 0.25) != 0:
            return False  # TODO be able to handle triplets

        note_len = note.quarterLength
        for _ in range(int(note_len / 0.25)):
            new_stream.append(m21.note.Note(note.pitch, quarterLength=0.25))

    return new_stream


def main():
    global FITNESS_THRESHOLD, POPULATION_SIZE, MAX_GENERATIONS, START_WITH_MARKOV

    for i in range(len(sys.argv)):
        if sys.argv[i] == '-f' or sys.argv[i] == '--desired-fitness':
            FITNESS_THRESHOLD = int(sys.argv[i + 1])
            i += 1
        elif sys.argv[i] == '-s' or sys.argv[i] == '--population-size':
            POPULATION_SIZE = int(sys.argv[i + 1])
            i += 1
        elif sys.argv[i] == '-g' or sys.argv[i] == '--max-generations':
            MAX_GENERATIONS = int(sys.argv[i + 1])
            i += 1
        elif sys.argv == '-m':
            START_WITH_MARKOV = True

    if SAVE_DATA:
        # set up a directory to save data to
        if not os.path.isdir("results"):
            os.mkdir("results")

        curr_time = str(time.time())

        save_dir = "results/" + curr_time  # the directory for a particular run will be named the current UNIX time, rounded down to the nearest second

        os.mkdir(save_dir)

        results = {"use_markov_to_generate_initial": START_WITH_MARKOV, "generation": [{"population": []}]}

    session, model = lstm.train_model_with_data()
    min_beats = 16
    max_beats = 16
    beats_per_measure = 4
    major = True

    score_titles = glob.glob('../corpus/*.mid')

    population = Queue()
    fitnesses = Queue()
    generations = 0

    # generate an initial population
    while fitnesses.qsize() < POPULATION_SIZE:
        if START_WITH_MARKOV:
            generated_score = markov.generate_melody(score_titles, random.randint(1, 3), random.randint(1, 3), min_beats, max_beats, beats_per_measure, major)
            score_cpy = copy.deepcopy(generated_score)
            converted_part = lstm.convert_part_to_sixteenth_notes(score_cpy.parts[0].notes)
        else:
            random_part = m21.stream.Part()

            for _ in range(min_beats * 4):
                random_part.append(m21.note.Note(m21.pitch.Pitch(random.randint(0, 127)), quarterLength=0.25))

            generated_score = m21.stream.Score()
            generated_score.append(random_part)
            score_cpy = copy.deepcopy(generated_score)
            converted_part = lstm.convert_part_to_sixteenth_notes(score_cpy[0].notes)

        if len(converted_part) != min_beats * 4:
            continue

        population.put(split_notes_into_sixteenth_notes(generated_score.parts[0].flat.notes))
        evaluation = lstm.evaluate_part(model, session, converted_part)[0]
        fitness = evaluation[0] - evaluation[1]
        fitnesses.put(fitness)

        print("Generated initial melody", fitnesses.qsize())

        if SAVE_DATA:
            file_name = curr_time + "_0_" + str(fitnesses.qsize() - 1) + ".mid"
            generated_score.write("midi", fp=save_dir + "/" + file_name)
            results["generation"][0]["population"].append({"fitness": str(fitness), "file_name": file_name})

    max_fitness = max(list(fitnesses.queue))
    average_fitness = sum(list(fitnesses.queue)) / fitnesses.qsize()
    min_fitness = min(list(fitnesses.queue))

    print(average_fitness, max_fitness, min_fitness)

    if SAVE_DATA:
        results["generation"][0]["average_fitness"] = str(average_fitness)

    generations += 1

    mutation_rate = 0.05

    while max_fitness < FITNESS_THRESHOLD and generations < MAX_GENERATIONS:
        print("Generation", generations)
        population_cpy = copy.deepcopy(population.queue)


        # remix the melodies to create new ones
        new_population = Queue()

        threads = []

        for i in range(4):
            thread = MutationThread(i, population, new_population)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # some crossover
        for _ in range(random.randint(0, int(POPULATION_SIZE / 4.0))):
            # pick the parents
            parent1_idx = int(random.triangular(0, POPULATION_SIZE, 0))
            parent2_idx = int(random.triangular(0, POPULATION_SIZE, 0))

            # choose a random number of crossover points
            child1, child2 = mutations.crossover(population_cpy[parent1_idx], population_cpy[parent2_idx], [int(random.triangular(2, 62)) for _ in range(int(random.triangular(0, 6)))])
            new_population.put(child1)
            new_population.put(child2)

        # mutate some stuff
        for melody in new_population.queue:
            # pick the indices to mutate
            mutate_indices = [random.randint(0, 63) for _ in range(int(random.triangular(0, 20, mutation_rate * 64)))]  # TODO change mutation rate over time

            for idx in mutate_indices:
                melody[idx].pitch.midi += random.randint(-5, 5)

        # evaluate the new population's fitness
        new_fitnesses = Queue()

        for new_melody in new_population.queue:
            new_melody_cpy = copy.deepcopy(new_melody)
            converted_part = lstm.convert_part_to_sixteenth_notes(new_melody_cpy.flat.notes)
            evaluation = lstm.evaluate_part(model, session, converted_part)[0]
            new_fitnesses.put(evaluation[0] - evaluation[1])

        # zipped_data = list(zip(new_fitnesses, new_population))
        # random.shuffle(zipped_data)
        # new_fitnesses, new_population = zip(*zipped_data)

        # maintain genetic diversity by randomly choosing which melodies to include, weighted toward the fitter melodies
        # fitnesses = []
        # population = []
        #
        # idx = 0
        # while len(fitnesses) < POPULATION_SIZE:
        #     rand_num = random.triangular(-5, 5, 3)
        #     if new_fitnesses[idx] > rand_num:
        #         fitnesses.append(new_fitnesses[idx])
        #         population.append(new_population[idx])
        #
        #     idx += 1
        #
        #     if len(fitnesses) - (idx - 1) + len(new_fitnesses) == POPULATION_SIZE:
        #         fitnesses += new_fitnesses[idx:]
        #         population += new_population[idx:]

        # fitnesses = new_fitnesses
        # population = new_population

        zipped_data = list(zip(new_fitnesses.queue, new_population.queue))
        zipped_data.sort(key=lambda x: x[0], reverse=True)
        newer_fitnesses, newer_population = list(zip(*zipped_data))

        new_fitnesses.queue = deque(newer_fitnesses)
        new_population.queue = deque(newer_population)

        fitnesses.queue = deque(list(new_fitnesses.queue)[:POPULATION_SIZE])
        population.queue = deque(list(new_population.queue)[:POPULATION_SIZE])

        max_fitness = max(list(fitnesses.queue))
        average_fitness = sum(list(fitnesses.queue)) / fitnesses.qsize()
        min_fitness = min(list(fitnesses.queue))

        print(average_fitness, max_fitness, min_fitness)

        if SAVE_DATA:
            results["generation"].append({"population": []})

            for idx, melody in enumerate(list(population.queue)):
                file_name = curr_time + "_" + str(generations) + "_" + str(idx) + ".mid"
                compact_notes(melody).write("midi", fp=save_dir + "/" + file_name)
                results["generation"][generations]["population"].append({"fitness": str(fitnesses.queue[idx]), "file_name": file_name})
                results["generation"][generations]["average_fitness"] = str(average_fitness)

        generations += 1

    if SAVE_DATA:
        with open(save_dir + "/results.json", "w") as fh:
            json.dump(results, fh)

    # view the entire last generation
    # for melody in population:
    #     compact_notes(melody).show()


if __name__ == '__main__':
    main()
