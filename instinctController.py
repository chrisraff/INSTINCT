import car
import numpy as np
from math import exp
from random import randint, random
from controllerUtils import getDistanceReadings, load_tracks
from fourierBasisController import FourierBasisController
from track import LapData
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from random import shuffle, seed
from track import *
from multiprocessing import Pool, cpu_count


track_glob = 'tracks_all/.'

training_generations = 10
pop_size = 8

mutation_std_decay = 1.5
min_mutation_std_dev = 0.01


seed(0) # shuffled track order will be the same across runs
np.random.seed(0) # random actions will be consistent run to run

class DNA():
    def __init__(self, arr):
        self.arr = arr

    def crossover(self, other_dna):
        # swap some parts
        my_w = self.arr
        your_w = other_dna.arr

        a = np.random.random( self.arr.shape ) > 0.5

        my_w *= a
        my_w += (1-a)*your_w

        self.arr = my_w
        return other_dna

    def mutate(self, curr_generation):
        # hit with some guassians

        # TODO tune this hyperparameter
        std_dev = (1-10**(-mutation_std_decay))**curr_generation
        std_dev = max(std_dev, min_mutation_std_dev)

        noise = np.random.normal(0, std_dev)

        self.arr += noise
        self.arr = np.abs(self.arr)


class InstinctController(FourierBasisController):

    def __init__(self, track, dna):
        super().__init__(track, degree=2)

        if dna is None:
            num_actions = 6
            num_percepts = self.numDistSensors+1+1  # plus one for speed, one for bias
            dna_w = np.zeros((num_percepts, num_actions))
            self.dna = DNA(dna_w)
        else:
            self.dna = dna


    def get_state_variables(self):
        return super().get_state_variables()


    def choose_action(self, state, eps=0):
        # return super().choose_action(state, eps)

        assert min(*state) >= 0

        if random() > eps:
            phi_s = self.fourier.phi(state)
            expected_returns = self.w @ phi_s  # size is (6,)

            # args = np.where( expected_returns == expected_returns.max() ) [0]
            # # assert args.shape[0] != 0
            # if args.shape[0] == 0:
            #     print(self.w)
            #     print(phi_s)
            #     print(expected_returns)
            #     print(expected_returns.max())
            # action = np.random.choice(args)
            # # print(expected_returns.reshape((2,3))); print(action)
            # return action #, expected_returns[args]
        else:
            expected_returns = np.zeros((6,))
            expected_returns[randint(0,5)] = 1
            # return randint(0,5)#, 0


        # softmax the expected returns
        expected_returns = np.exp(expected_returns) / np.sum(np.exp(expected_returns))


        # TODO maybe do this a different way. like averaging :)
        instinct_expected_returns = (state+[1]) @ self.dna.arr


        expected_returns_with_instincts_accounted_for = np.maximum(expected_returns, instinct_expected_returns)


        # args = np.where( expected_returns == expected_returns.max() ) [0]
        # # assert args.shape[0] != 0
        # if args.shape[0] == 0:
        #     print(self.w)
        #     print(phi_s)
        #     print(expected_returns)
        #     print(expected_returns.max())
        # action = np.random.choice(args)
        # # print(expected_returns.reshape((2,3))); print(action)
        # return action #, expected_returns[args]


        return expected_returns_with_instincts_accounted_for.argmax()


    def update(self):
        return super().update()


    def reset_and_punish(self):
        super().reset_and_punish()


    def update_track(self, track):
        super().update_track(track)



def train(agent):
    agent.train = True
    agent.auto_reset = True

    while True:
        result = agent.update()
        if agent.car.lapData.nextCheckpoint == agent.car.lapData.numCheckpoints-1:
            print("wow, it ran a whole track!")
            return agent
        if result == FourierBasisController.UPDATERESULT_RESET:
            return agent



class Population:
    def __init__(self):
        # hyperparameters
        self.training_generations = training_generations
        self.pop_size = pop_size

        self.curr_generation = 0
        self.pop = self.make_population()

        # load stuff
        print('loading tracks')
        self.tracks = load_tracks(track_glob, tqdm)

        print('building track lines')
        for track in tqdm(self.tracks):
            track.updateTrackLines()

        pass


    def make_population(self):
        return [ InstinctController(Track(), dna=None) for _ in range(self.pop_size) ]


    def evaluate_agents(self):
        print("EVALUATING gen {}".format(self.curr_generation))
        # TODO pick the track everyone will be training on randomly instead (according to a seed)
        curr_track = self.tracks[self.curr_generation % len(self.tracks)]

        # reset the agents and plop them into their latest fun little track!
        for agent in self.pop:
            agent.update_track( curr_track )
            agent.epsilon = 0.001


        # # without threading
        # self.pop = [train(agent) for agent in tqdm(self.pop)]

        # with threading
        with Pool(cpu_count()) as p:
            self.pop = list(tqdm(p.imap(train, self.pop), total=self.pop_size))

        self.curr_generation += 1


    def breed_next_generation_agents(self):
        print("BREEDING gen {}".format(self.curr_generation))
        # fitness!

        # for agent in self.pop:
        #     fitness = agent.returns[-1]
        fitness_and_agents = [(agent.returns[-1], agent) for agent in self.pop]

        fitnesses = np.array([agent.returns[-1] for agent in self.pop])
        fitnesses = np.exp(fitnesses) / np.sum(np.exp(fitnesses))

        new_pop = []
        for i in range(self.pop_size):
            sample_dad = np.random.choice(self.pop, p=fitnesses)
            sample_mom = np.random.choice(self.pop, p=fitnesses)

            kid_dna = sample_dad.dna.crossover(sample_mom.dna)
            kid_dna.mutate(self.curr_generation)
            kid = InstinctController(Track(), dna=kid_dna)

            new_pop += [ kid ]

        self.pop = new_pop

    def get_champion(self):
        return max(self.pop, key=lambda x: x.returns[-1])


def main():
    pop_object = Population()
    print("training population")
    while pop_object.curr_generation < pop_object.training_generations-1:
        pop_object.evaluate_agents()
        pop_object.breed_next_generation_agents()
    pop_object.evaluate_agents()


    # pickle the population
    print("pickling the population")
    pop_fname = "population.pickle"
    with open(pop_fname , 'wb') as f:
        pickle.dump(pop_object, f)


if __name__ == "__main__":
    main()
