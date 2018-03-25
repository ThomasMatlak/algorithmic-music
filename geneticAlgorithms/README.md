# Genetic Algorithms #
Generate melodies using a genetic algorithm.
Creates a random population, remixes and mutates it, evaluates the fitness of melodies using long-short term memory artificial neural network, and selects the best melodies to seed the next generation.

## How to Use ##
As an example, to run the program with a population size of 100 for 500 generations, with a desired fitness of 0.9, use

`python3 geneticAlgorithms.py -s 100 -g 500 -f 0.9`

You may also have the program use an initial population generated using Markov chains by adding the `-m` flag to the command.

Note that this program requires the use of the [`multiprocess`](https://github.com/uqfoundation/multiprocess) library.

