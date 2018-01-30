#!/usr/bin/python3

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


POPULATION_SIZE = 10
MAX_GENERATIONS = 10
FITNESS_THRESHOLD = 5  # fitness is measured as the difference between the LSTM NN's scores of good and bad
SAVE_DATA = True


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

        results = {"generation": [{"population": []}]}

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
        generated_score = markov.generate_melody(score_titles, random.randint(1, 3), random.randint(1, 3), min_beats, max_beats, beats_per_measure, major)
        score_cpy = copy.deepcopy(generated_score)
        converted_part = lstm.convert_part_to_sixteenth_notes(score_cpy.parts[0].notes)

        if len(converted_part) != 64:
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

    print(fitnesses)

    generations += 1

    while max(fitnesses) < FITNESS_THRESHOLD and generations < MAX_GENERATIONS:
        print("Generation", generations)

        # remix the melodies to create new ones
        new_population = []

        for melody in population:
            for i in range(0, 13):
                new_population.append(mutations.transpose(melody, i))
            new_population.append(mutations.inverse(melody))
            new_population.append(mutations.inverse_retrograde(melody))
            new_population.append(mutations.retrograde_inverse(melody))
            new_population.append(mutations.retrograde(melody, True, False))
            new_population.append(mutations.retrograde(melody, False, True))
            new_population.append(mutations.retrograde(melody, True, True))

        # perform a random number of crossovers and remix the crossovers
        for _ in range(random.randrange(10)):
            parent1 = random.choice(new_population)
            parent2 = random.choice(new_population)

            for _ in range(random.randrange(10)):
                child1, child2 = mutations.crossover(parent1, parent2, [random.randrange(2, 62) for _ in range(random.randrange(4))])

                for i in range(0, 13):
                    new_population.append(mutations.transpose(child1, i))
                new_population.append(mutations.inverse(child1))
                new_population.append(mutations.inverse_retrograde(child1))
                new_population.append(mutations.retrograde_inverse(child1))
                new_population.append(mutations.retrograde(child1, True, False))
                new_population.append(mutations.retrograde(child1, False, True))
                new_population.append(mutations.retrograde(child1, True, True))

                for i in range(0, 13):
                    new_population.append(mutations.transpose(child2, i))
                new_population.append(mutations.inverse(child2))
                new_population.append(mutations.inverse_retrograde(child2))
                new_population.append(mutations.retrograde_inverse(child2))
                new_population.append(mutations.retrograde(child2, True, False))
                new_population.append(mutations.retrograde(child2, False, True))
                new_population.append(mutations.retrograde(child2, True, True))

        # Introduce some random changes

        # evaluate the new population's fitness
        new_fitnesses = []

        for new_melody in new_population:
            new_melody_cpy = copy.deepcopy(new_melody)
            converted_part = lstm.convert_part_to_sixteenth_notes(new_melody_cpy.flat.notes)
            evaluation = lstm.evaluate_part(model, session, converted_part)[0]
            new_fitnesses.append(evaluation[0] - evaluation[1])

        # take the top 10 melodies to reproduce
        zipped_data = list(zip(new_fitnesses, new_population))
        zipped_data.sort(key=lambda x: x[0], reverse=True)
        new_fitnesses, new_population = zip(*zipped_data)

        fitnesses = new_fitnesses[:POPULATION_SIZE]
        population = new_population[:POPULATION_SIZE]

        print(fitnesses)

        if SAVE_DATA:
            results["generation"].append({"population": []})

            for idx, melody in enumerate(population):
                file_name = curr_time + "_" + str(generations) + "_" + str(idx) + ".mid"
                melody.write("midi", fp=save_dir + "/" + file_name)
                results["generation"][generations]["population"].append({"fitness": str(fitnesses[idx]), "file_name": file_name})

        generations += 1

    if SAVE_DATA:
        with open(save_dir + "/results.json", "w") as fh:
            json.dump(results, fh)

    # view the entire last generation
    for melody in population:
        compact_notes(melody).show()


if __name__ == '__main__':
    main()
