import numpy as np
from fourierBasisController import FourierBasisController
from tqdm import tqdm, trange
from controllerUtils import *

'''
DONE - generate some test tracks

run the fourier controller on the test tracks
record the return for every test track

run the champion instinct controller on the test tracks
record the return for every test track
'''

# load all test tracks
print('loading tracks')
track_dir = 'tracks_test'
test_tracks = load_tracks(track_dir, tqdm)



# load fourier controller
print("loading fourier controller")
fourier_controller = pickle.load(open('fourierController.pickle', 'rb'))

# load instinct controller
print("loading instinct controller")
instinct_controller = pickle.load(open('champion.pickle', 'rb'))


def get_return_for_track(track, agent):
    agent.update_track( track )
    agent.epsilon = 0.001


    agent.train = False
    agent.auto_reset = True

    while True:
        result = agent.update()
        if agent.car.lapData.nextCheckpoint == agent.car.lapData.numCheckpoints-1:
            # TODO mark this as having a big fitness
            # TODO implement this in the fourier controller
            agent.returns += [agent.current_return]
            return agent
        if result == FourierBasisController.UPDATERESULT_RESET:
            return agent

    # ONLY DO THIS FOR THE INSTINCT CONTROLLER!
    # ONLY DO THIS FOR THE INSTINCT CONTROLLER!
    # ONLY DO THIS FOR THE INSTINCT CONTROLLER!
    agent.w = np.zeros_like(agent.w)
    # ONLY DO THIS FOR THE INSTINCT CONTROLLER!
    # ONLY DO THIS FOR THE INSTINCT CONTROLLER!
    # ONLY DO THIS FOR THE INSTINCT CONTROLLER!

    return agent.returns[-1]


def main():
    fourier_returns = []
    instinct_returns = []
    for track in test_tracks:
        fourier_return = get_return_for_track(track, fourier_controller)
        fourier_returns += [fourier_return]

        instinct_return = get_return_for_track(track, instinct_controller)
        instinct_returns += [instinct_return]

    # plot stuff



if __name__ == "__main__":
    main()
