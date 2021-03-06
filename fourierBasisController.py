import car
import numpy as np
from math import exp
from random import randint, random
from controllerUtils import getDistanceReadings, load_tracks
from FourierBasis import *
from track import LapData
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle, seed


training_episodes = 20000
track_glob = 'tracks_all/.'

seed(0) # shuffled track order will be the same across runs
np.random.seed(0) # random actions will be consistent run to run


class FourierBasisController:

    UPDATERESULT_NOTHING = 0
    UPDATERESULT_RESET = 1



    def __init__(self, track, degree=2):
        self.car = car.Car(track.start_position[0], track.start_position[1], track)
        self.car.dir = track.start_direction
        self.track = track

        # controller variables go here
        self.numDistSensors = 5

        self.fourier = FourierBasis(self.numDistSensors + 1, degree, approximate_upper_bound=100)
        self.w = np.zeros( (6, self.fourier.c_vec.shape[0]) ) # 6 actions

        self.car.initLapData()

        self.percepts = self.get_state_variables()

        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.001

        self.step_size = 1e-10

        self.action = 0
        self.frames_per_action = 5
        self.frames_this_action = self.frames_per_action
        self.reward_this_action = 0 # reward_this_action is currently unused in favor of lap progress
        self.progress_at_start_of_action = 0

        self.returns = []
        self.current_return = 0
        self.current_discount = 1

        self.actions_per_episode = [0]
        self.checkpoints_per_episode = [0]

        self.train = True
        self.auto_reset = True

        self.last_td_error = 0

        self.checkpoint_reward_strength = 60
        self.lap_progress_reward_strength = 60
        self.death_reward_strength = -10


    def get_state_variables(self):
        percepts = getDistanceReadings(self.car, self.track, self.numDistSensors)

        percepts += [self.car.speed]

        return percepts


    # selects an action given a state using the state approximator, or choosing randomly with probability epsilon
    def choose_action(self, state, eps=0):
        assert min(*state) >= 0

        if random() > eps:
            phi_s = self.fourier.phi(state)
            expected_returns = self.w @ phi_s

            args = np.where( expected_returns == expected_returns.max() ) [0]
            # assert args.shape[0] != 0
            if args.shape[0] == 0:
                print(self.w)
                print(phi_s)
                print(expected_returns)
                print(expected_returns.max())
            action = np.random.choice(args)
            # print(expected_returns.reshape((2,3))); print(action)
            return action #, expected_returns[args]
        else:
            return randint(0,5)#, 0


    def update(self):
        # controller logic goes here

        if (self.frames_per_action <= self.frames_this_action):
            self.actions_per_episode[-1] += 1

            s_new = self.get_state_variables()

            # reward = self.reward_this_action
            current_progress = self.car.lapData.getProgress()
            reward = (current_progress - self.progress_at_start_of_action) * self.lap_progress_reward_strength
            # print(reward)
            self.progress_at_start_of_action = current_progress

            if self.train:
                # Reward the previous state
                phi_s = self.fourier.phi(self.percepts)

                phi_s_new = self.fourier.phi(s_new)
                action_greedy = self.choose_action(s_new, eps=0)

                expected_next_return = phi_s_new @ self.w[action_greedy]

                # td update
                td_error = (reward + self.gamma * expected_next_return - (phi_s @ self.w[self.action]))
                update = self.step_size * td_error * phi_s
                self.last_td_error = td_error

                self.w[self.action] += update

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.current_return += self.current_discount * reward

            # Get the new action
            self.percepts = s_new

            self.action = self.choose_action(self.percepts, self.epsilon)

            self.frames_this_action = 0
            self.reward_this_action = 0 # reward_this_action is currently unused in favor of lap progress
            self.current_discount *= self.gamma

        self.frames_this_action += 1

        # turn the action [0...5] into car inputs [brk lft, brk cen, brk rit, gas lft, gas cen, gas rit]
        steering = (self.action % 3 - 1) * (self.car.getSteeringThreshold() - 0.01)
        thrust   = -1 if self.action < 3 else 1

        self.car.steering =  steering
        self.car.throttle =  thrust if thrust > 0 else 0
        self.car.brake    = -thrust if thrust < 0 else 0

        # gets the number of checkpoints passed this frame in case it is used for rewards
        result = self.car.update()
        self.checkpoints_per_episode[-1] += result

        self.reward_this_action += result * self.checkpoint_reward_strength # reward_this_action is currently unused in favor of lap progress

        # the agent can reset itself in terminal situations
        if self.auto_reset and (self.car.offRoad or (self.car.speed == 0 and thrust == -1 and self.frames_this_action == 1)):
            self.reset_and_punish()

            return FourierBasisController.UPDATERESULT_RESET

        return FourierBasisController.UPDATERESULT_NOTHING


    def reset_and_punish(self):
        if self.train:
            # punish the agent
            reward = self.death_reward_strength
            phi_s = self.fourier.phi(self.percepts)

            # td update - the episode has ended so do not include the next estimation
            td_error = (reward - (phi_s @ self.w[self.action]))
            update = self.step_size * td_error * phi_s
            self.last_td_error = td_error

            self.w[self.action] += update

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # reset the car
        self.car = car.Car(self.track.start_position[0], self.track.start_position[1], self.track)
        self.car.dir = self.track.start_direction
        self.car.initLapData()

        self.percepts = self.get_state_variables()

        self.current_return += self.current_discount * (self.death_reward_strength)

        self.frames_this_action = self.frames_per_action
        self.returns += [self.current_return]
        self.reward_this_action = 0 # reward_this_action is currently unused in favor of lap progress
        self.progress_at_start_of_action = self.car.lapData.getProgress()
        self.current_return = 0
        self.current_discount = 1

        self.actions_per_episode += [0]
        self.checkpoints_per_episode += [0]


    def update_track(self, track):
        self.track = track

        # reset the car
        self.car = car.Car(self.track.start_position[0], self.track.start_position[1], self.track)
        self.car.dir = self.track.start_direction
        self.car.initLapData()

        self.percepts = self.get_state_variables()

        self.frames_this_action = self.frames_per_action
        self.reward_this_action = 0 # reward_this_action is currently unused in favor of lap progress
        self.progress_at_start_of_action = self.car.lapData.getProgress()
        self.current_return = 0
        self.current_discount = 1


def train(fourierAgent, episodes, tracks, save_fname=None, save_every=100):
    fourierAgent.train = True
    fourierAgent.auto_reset = True

    ep = 0
    tracks_queue = [] # this will get populated in the while loop

    with tqdm(total=episodes) as pbar:
        while ep < episodes:

            result = fourierAgent.update()

            if result == FourierBasisController.UPDATERESULT_RESET:
                pbar.update()
                ep += 1

                # change the track
                if len(tracks_queue) == 0:
                    tracks_queue = tracks[:]
                    shuffle(tracks_queue)

                next_track = tracks_queue.pop()
                fourierAgent.update_track( next_track )

                # save the agent intermittently
                if save_fname is not None and ep % save_every == 0:
                    with open(save_fname , 'wb') as f:
                        pickle.dump(fourierAgent, f)

    # save the agent once done
    if save_fname is not None:
        with open(save_fname , 'wb') as f:
            pickle.dump(fourierAgent, f)


def main():
    print('loading tracks')
    tracks = load_tracks(track_glob, tqdm)

    # print('building track lines')
    # for track in tqdm(tracks):
    #     track.updateTrackLines()

    print('training agent')

    agent = FourierBasisController(tracks[0])

    train(agent, training_episodes, tracks, 'fourierController.pickle', 100)

    plt.plot(agent.returns, 'o')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

    plt.plot(agent.actions_per_episode[:-1], 'o')
    plt.xlabel('Episode')
    plt.ylabel('Number of Actions')
    plt.show()

    plt.plot(agent.checkpoints_per_episode[:-1], 'o')
    plt.xlabel('Episode')
    plt.ylabel('Number of Checkpoints')
    plt.show()


if __name__ == "__main__":
    main()
