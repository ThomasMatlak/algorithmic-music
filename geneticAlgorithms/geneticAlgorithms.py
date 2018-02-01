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


POPULATION_SIZE = 25
MAX_GENERATIONS = 50
FITNESS_THRESHOLD = 5  # fitness is measured as the difference between the LSTM NN's scores of good and bad
SAVE_DATA = True
START_WITH_MARKOV = True


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

    population = []
    fitnesses = []
    generations = 0

    # generate an initial population
    while len(fitnesses) < POPULATION_SIZE:
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

        population.append(split_notes_into_sixteenth_notes(generated_score.parts[0].flat.notes))
        evaluation = lstm.evaluate_part(model, session, converted_part)[0]
        fitness = evaluation[0] - evaluation[1]
        fitnesses.append(fitness)

        print("Generated initial melody", len(fitnesses))

        if SAVE_DATA:
            file_name = curr_time + "_0_" + str(len(fitnesses) - 1) + ".mid"
            generated_score.write("midi", fp=save_dir + "/" + file_name)
            results["generation"][0]["population"].append({"fitness": str(fitness), "file_name": file_name})

    max_fitness = max(fitnesses)
    average_fitness = sum(fitnesses) / len(fitnesses)
    min_fitness = min(fitnesses)

    print(average_fitness, max_fitness, min_fitness)

    if SAVE_DATA:
        results["generation"][0]["average_fitness"] = str(average_fitness)

    generations += 1

    mutation_rate = 0.1

    while max(fitnesses) < FITNESS_THRESHOLD and generations < MAX_GENERATIONS:
        print("Generation", generations)

        # remix the melodies to create new ones
        new_population = []

        for melody in population:
            for i in range(0, 13):
                if random.random() < 0.8:
                    new_population.append(mutations.transpose(melody, i))

            if random.random() < 0.7:
                new_population.append(mutations.inverse(melody))
            if random.random() < 0.5:  # inverse-retrograde is causing problems
                new_population.append(mutations.inverse_retrograde(melody))
            if random.random() < 0.5:
                new_population.append(mutations.retrograde_inverse(melody))
            if random.random() < 0.5:
                new_population.append(mutations.retrograde(melody, True, True))
            if random.random() < 0.5:
                new_population.append(mutations.retrograde(melody, True, False))
            if random.random() < 0.5:
                new_population.append(mutations.retrograde(melody, False, True))

        # some crossover
        for _ in range(random.randint(0, int(POPULATION_SIZE / 4.0))):
            # pick the parents
            parent1_idx = int(random.triangular(0, POPULATION_SIZE, 0))
            parent2_idx = int(random.triangular(0, POPULATION_SIZE, 0))

            # choose a random number of crossover points
            child1, child2 = mutations.crossover(population[parent1_idx], population[parent2_idx], [int(random.triangular(2, 62)) for _ in range(int(random.triangular(0, 6)))])
            new_population.append(child1)
            new_population.append(child2)

        # mutate some stuff
        for melody in new_population:
            # pick the indices to mutate
            mutate_indices = [random.randint(0, 63) for _ in range(int(random.triangular(0, 20, mutation_rate * 64)))]

            for idx in mutate_indices:
                melody[idx].pitch.midi += random.randint(-5, 5)

        # evaluate the new population's fitness
        new_fitnesses = []

        for new_melody in new_population:
            new_melody_cpy = copy.deepcopy(new_melody)
            converted_part = lstm.convert_part_to_sixteenth_notes(new_melody_cpy.flat.notes)
            evaluation = lstm.evaluate_part(model, session, converted_part)[0]
            new_fitnesses.append(evaluation[0] - evaluation[1])

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

        fitnesses = new_fitnesses
        population = new_population

        zipped_data = list(zip(fitnesses, population))
        # random.shuffle(zipped_data)
        zipped_data.sort(key=lambda x: x[0], reverse=True)
        fitnesses, population = zip(*zipped_data)

        fitnesses = fitnesses[:POPULATION_SIZE]
        population = population[:POPULATION_SIZE]

        max_fitness = max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)
        min_fitness = min(fitnesses)

        print(average_fitness, max_fitness, min_fitness)

        if SAVE_DATA:
            results["generation"].append({"population": []})

            for idx, melody in enumerate(population):
                file_name = curr_time + "_" + str(generations) + "_" + str(idx) + ".mid"
                compact_notes(melody).write("midi", fp=save_dir + "/" + file_name)
                results["generation"][generations]["population"].append({"fitness": str(fitnesses[idx]), "file_name": file_name})
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
