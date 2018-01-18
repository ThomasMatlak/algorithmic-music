#!/usr/bin/python3
import intervalMarkovChain as markov
import lstm
import music21 as m21
import glob
import mutations


def main():
    session, model = lstm.train_model_with_data()

    min_beats = 16
    max_beats = 16
    beats_per_measure = 4
    major = True

    score_titles = glob.glob('../corpus/*.mid')

    population = []
    fitnesses = []

    # generate an initial population

    while len(fitnesses) < 10:
        generated_score = markov.generate_melody(score_titles, 3, 2, min_beats, max_beats, beats_per_measure, major)
        converted_part = lstm.convert_part(generated_score.parts[0])

        if len(converted_part) != 64:
            continue

        fitness = lstm.evaluate_part(model, session, converted_part[:64])[0][0]

        population.append(generated_score)
        fitnesses.append(fitness)

    print(fitnesses)


if __name__ == '__main__':
    main()
