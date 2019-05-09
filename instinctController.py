from time import time
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
from random import choice
import argparse


# default values if you don't use the arguments
track_glob = 'tracks_all/'
pickle_champion_every_n_generations = 1
training_generations = 20#30
pop_size = 20
num_elites = 6
num_purges = 1
sigma = 12  # parameter for softmax that turns agent fitnesses into breeding probabilities
mutation_std_decay = 1.0
min_mutation_std_dev = 0.01
tracks_per_generation = 8


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--track_glob", type=str, default=track_glob)
parser.add_argument("-c", "--pickle_champion_every_n_generations", type=int, default=pickle_champion_every_n_generations)
parser.add_argument("-t", "--training_generations", type=int, default=training_generations)
parser.add_argument("-p", "--pop_size", type=int, default=pop_size)
parser.add_argument("-e", "--num_elites", type=int, default=num_elites)
parser.add_argument("-n", "--num_purges", type=int, default=num_purges)
parser.add_argument("-s", "--sigma", type=float, default=sigma)
parser.add_argument("-m", "--mutation_std_decay", type=float, default=mutation_std_decay)
parser.add_argument("-i", "--min_mutation_std_dev", type=float, default=min_mutation_std_dev)
parser.add_argument("-g", "--tracks_per_generation", type=int, default=tracks_per_generation)

args = parser.parse_args()

track_glob = args.track_glob
pickle_champion_every_n_generations = args.pickle_champion_every_n_generations
training_generations = args.training_generations
pop_size = args.pop_size
num_elites = args.num_elites
num_purges = args.num_purges
sigma = args.sigma
mutation_std_decay = args.mutation_std_decay
min_mutation_std_dev = args.min_mutation_std_dev
tracks_per_generation = args.tracks_per_generation




seed(0) # shuffled track order will be the same across runs
np.random.seed(0) # random actions will be consistent run to run


def softmax(expected_returns, s=1):
    exps = np.exp(s * (expected_returns - np.max(expected_returns)))
    exps /= np.sum(exps)
    return exps


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

        noise = np.random.normal(0, std_dev, self.arr.shape)

        self.arr += noise


class InstinctController(FourierBasisController):

    def __init__(self, track, dna):
        super().__init__(track, degree=2)

        self.hyperparameters = {
            'track_glob':track_glob,
            'pickle_champion_every_n_generations':pickle_champion_every_n_generations,
            'training_generations':training_generations,
            'pop_size':pop_size,
            'num_elites':num_elites,
            'num_purges':num_purges,
            'sigma':sigma,
            'mutation_std_decay':mutation_std_decay,
            'min_mutation_std_dev':min_mutation_std_dev,
            'tracks_per_generation':tracks_per_generation,
        }

        self.actions_so_far = 0
        self.times_instinct_took_action = 0

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


        # softmax the experience expected returns
        expected_returns = softmax(expected_returns)


        # multiply the weights to determine what the instinct part wants to do
        instinct_expected_returns = (state+[1]) @ self.dna.arr

        # softmax the instinct returns
        instinct_expected_returns = softmax(instinct_expected_returns)


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

        if True:
        # if self.train:
        #     return instinct_expected_returns.argmax()
        # else:
            thing = expected_returns_with_instincts_accounted_for.argmax()
            instinct_took_action = bool(np.argmax(  [ expected_returns[thing], instinct_expected_returns[thing] ]  ))

            self.actions_so_far += 1
            self.times_instinct_took_action += int(instinct_took_action)

            if not self.train:
                # print the percentage of time we follow the advice of the instinct controller
                print("{:.3%} {}".format(self.times_instinct_took_action/self.actions_so_far, "INSTINCT" if instinct_took_action else "EXPERIENCE"))
                # print("exp:{} INS:{} max:{}".format(expected_returns, instinct_expected_returns, expected_returns_with_instincts_accounted_for))

            return thing


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
            # TODO mark this as having a big fitness
            # TODO implement this in the fourier controller
            print("wow, it ran a whole track!")
            agent.returns += [agent.current_return+5]
            return agent
        if result == FourierBasisController.UPDATERESULT_RESET:
            return agent



class Population:
    def __init__(self):
        # hyperparameters
        self.training_generations = training_generations
        self.pop_size = pop_size

        self.hyperparameters = {
            'track_glob':track_glob,
            'pickle_champion_every_n_generations':pickle_champion_every_n_generations,
            'training_generations':training_generations,
            'pop_size':pop_size,
            'num_elites':num_elites,
            'num_purges':num_purges,
            'sigma':sigma,
            'mutation_std_decay':mutation_std_decay,
            'min_mutation_std_dev':min_mutation_std_dev,
            'tracks_per_generation':tracks_per_generation,
        }

        self.curr_generation = 0
        self.pop = self.make_population()

        self.top_fitnesses_by_generation = []

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
        print("EVALUATING gen {}/{}".format(self.curr_generation+1, self.training_generations))

        # # TODO pick the track everyone will be training on randomly instead (according to a seed)
        # curr_track = self.tracks[self.curr_generation % len(self.tracks)]

        tracks_to_run = [choice(self.tracks) for _ in range(tracks_per_generation)]

        for curr_track in tqdm(tracks_to_run):

            # reset the agents and plop them into their latest fun little track!
            for agent in self.pop:
                # curr_track = choice(self.tracks)

                agent.update_track( curr_track )
                agent.epsilon = 0.001

            # # without threading
            # self.pop = [train(agent) for agent in tqdm(self.pop)]

            # with threading
            with Pool(cpu_count()) as p:
                # self.pop = list(tqdm(p.imap(train, self.pop), total=self.pop_size))
                self.pop = list(p.imap(train, self.pop))  #without tqdm

            # reset the experience weights so we don't get an unfair advantage when running more trials
            for agent in self.pop:
                agent.w = np.zeros_like(agent.w)

        def fitness(x):
            # return np.mean(x)
            # return np.min(x)
            return np.mean([np.mean(x), np.min(x)])


        # sort the population by fitness
        self.pop = sorted(self.pop, key=lambda x: fitness(x.returns), reverse=True)




        fitnesses = [fitness(agent.returns) for agent in self.pop] # changed this to means, not most recent
        top1, top2, top3 = fitnesses[:3]
        print("fitness: top: {:.4f} {:.4f} {:.4f} median: {:.4f} avg: {:.4f} min: {:.4f}".format( top1, top2, top3, fitnesses[self.pop_size//2], np.mean(fitnesses), fitnesses[-1] ))

        self.top_fitnesses_by_generation += [top1]

        self.curr_generation += 1


    def breed_next_generation_agents(self):
        print("BREEDING gen {}/{}".format(self.curr_generation+1, self.training_generations))

        # TODO remove the `num_purges` worst performing agents
        if num_purges > 0:
            self.pop = self.pop[:-num_purges]

        fitnesses = np.array([agent.returns[-1] for agent in self.pop])
        fitnesses = softmax(fitnesses, sigma)

        top_agents = self.pop[:num_elites]

        new_pop = []
        new_pop += top_agents
        for i in range(self.pop_size-num_elites):
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
    print("training population")
    start_time = time()
    pop_object = Population()
    while pop_object.curr_generation < pop_object.training_generations-1:
        pop_object.evaluate_agents()

        if pop_object.curr_generation % pickle_champion_every_n_generations == 0:
            # pickle the champion
            print("pickling the champion")
            pop_fname = "champion.pickle"
            with open(pop_fname , 'wb') as f:
                pickle.dump(pop_object.get_champion(), f)

        pop_object.breed_next_generation_agents()


    pop_object.evaluate_agents()
    duration1 = time()-start_time
    print("it took {:.3f} seconds to run {} generations with a population of {} with {} elites".format(duration1, training_generations, pop_size, num_elites))

    print('Top fitness of each gen:')
    print(pop_object.top_fitnesses_by_generation)

    # pickle the champion
    print("pickling the champion")
    start_time = time()
    pop_fname = "champion{}.pickle".format(time())
    with open(pop_fname , 'wb') as f:
        pickle.dump(pop_object.get_champion(), f)
    duration2 = time()-start_time
    print("it took {:.3f} seconds to pickle the champion".format(duration2))

    # pickle the population
    print("pickling the population")
    start_time = time()
    pop_fname = "population{}.pickle".format(time())
    with open(pop_fname , 'wb') as f:
        pickle.dump(pop_object, f)
    duration3 = time()-start_time
    print("it took {:.3f} seconds to pickle the population".format(duration3))

    print("total time was {:.3f} seconds".format(duration1+duration2+duration3))


if __name__ == "__main__":
    main()
