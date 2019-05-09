from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count
import numpy as np
from track import *
from fourierBasisController import *
from straightController import *
from instinctController import *
from initInstinctController import *
from tqdm import tqdm, trange
from controllerUtils import *

'''
DONE - generate some test tracks

run the fourier controller on the test tracks
record the return for every test track

run the champion instinct controller on the test tracks
record the return for every test track
'''


# if the agent runs this many laps, stop the simulation so it doesn't run forever
max_laps_per_track = 2


# load fourier controller
print("loading fourier controller")
fourier_controller = pickle.load(open('fourierController.pickle', 'rb'))

# load instinct controller
print("loading instinct controller")
# instinct_controller = pickle.load(open('awesome_champion.pickle', 'rb'))
# instinct_controller = pickle.load(open('champion1557358749.1494272.pickle', 'rb'))
# instinct_controller = pickle.load(open('champion1557370035.5170279.pickle', 'rb'))
# instinct_controller = pickle.load(open('champion.pickle', 'rb'))
instinct_controller = pickle.load(open('championInstinct1557430173.3814373.pickle', 'rb')) #chris initial guess good controller
instinct_controller.actions_so_far = 0
instinct_controller.times_instinct_took_action = 0

# load init controller
print("loading init controller")
init_controller = pickle.load(open('champion1557406890.5402772.pickle', 'rb'))

# load all test tracks
print('loading tracks')
# tracks_dir = 'tracks' #train
tracks_dir = 'tracks_test' #test
tracks = load_tracks(tracks_dir, tqdm)



def get_return_for_track(track, agent, reset_experience):
    # this needs to happen when testing the instinct agent
    if reset_experience:
        agent.w = np.zeros_like(agent.w)
        agent.epsilon = 0.001

    agent.update_track( track )


    agent.train = False
    agent.auto_reset = True

    while True:
        result = agent.update()
        if agent.car.lapData.nextCheckpoint == agent.car.lapData.numCheckpoints-1:
            # TODO mark this as having a big fitness
            # TODO implement this in the fourier controller
            # print("wow, it ran a whole track!")
            agent.returns += [agent.current_return]
            return agent.returns[-1]
        if result == FourierBasisController.UPDATERESULT_RESET:
            return agent.returns[-1]



    # # this needs to happen when testing the instinct agent
    # if reset_experience:
    #     agent.w = np.zeros_like(agent.w)
    #     agent.epsilon = 0.001

    # agent.update_track( track )

    # agent.train = False
    # agent.auto_reset = True

    # checkpoints = 0
    # laps = 0
    # while True:
    #     result = agent.update()
    #     checkpoints = agent.car.lapData.nextCheckpoint
    #     if agent.car.lapData.nextCheckpoint == agent.car.lapData.numCheckpoints-1:
    #         laps += 1
    #     if laps == max_laps_per_track:
    #         # TODO mark this as having a big fitness
    #         # TODO implement this in the fourier controller
    #         agent.returns += [agent.current_return]
    #         print('great ' + str(agent.returns))
    #         break
    #     if result == FourierBasisController.UPDATERESULT_RESET:
    #         print('bad ' + str(agent.returns))
    #         break

    # # return checkpoints
    # return agent.returns[-1]


def multiprocessing_get_return(data):
    # print(data)
    track, agents, reset_experiences = data

    # return a list of the three agents' returns for this track
    return [get_return_for_track(track, agents[i], reset_experiences[i]) for i in range(len(agents))]


def plot_returns(returns, label):
    if len(returns) == 0:
        print("there weren't no returns for {}, not plotting it".format(label))
        return

    # plt.plot(np.arange(len(returns)), returns, marker='o', alpha=0.8)
    plt.scatter(np.arange(len(returns)), returns, alpha=0.7, label=label)



if __name__ == "__main__":
    assert type(fourier_controller) == FourierBasisController, "the instinct controller is actually a "+str(type(instinct_controller))
    assert type(instinct_controller) == InstinctController, "the instinct controller is actually a "+str(type(instinct_controller))
    assert type(init_controller) == InitInstinctController, "the instinct controller is actually a "+str(type(instinct_controller))



    print("running the top agents on the test tracks (threaded)")
    num_tracks = len(tracks)
    agents = (fourier_controller, instinct_controller, init_controller)
    reset_experiences = (True, False, False)
    with Pool(cpu_count()) as p:
        all_returns = list(tqdm(p.imap(multiprocessing_get_return, zip(tracks, [agents]*num_tracks, [reset_experiences]*num_tracks)), total=num_tracks))
        # all_returns = list(p.imap(multiprocessing_get_return, zip(tracks, agents, reset_experiences)))  #without tqdm
    all_returns = sorted(all_returns, key=lambda x: x[0])
    # all_returns = sorted(all_returns, key=lambda x: np.mean(x))
    fourier_returns, instinct_returns, init_returns = zip(*all_returns)


    print("running the top agents on the test tracks (unthreaded)")
    # fourier_returns = []
    # instinct_returns = []
    # init_returns = []
    # # for each test track
    # # for i, track in tqdm(enumerate(tracks)):
    # for i, track in enumerate(tracks):
    #     print("track index: "+str(i))
    #     instinct_returns += [get_return_for_track(track, instinct_controller, reset_experience=True)]
    #     # fourier_returns += [get_return_for_track(track, fourier_controller, reset_experience=False)]
    #     # init_returns += [get_return_for_track(track, init_controller, reset_experience=False)]

    # all_returns = zip(fourier_returns, instinct_returns, init_returns)
    # all_returns = sorted(all_returns, key=lambda x: x[0])
    # # all_returns = sorted(all_returns, key=lambda x: np.mean(x))
    # fourier_returns, instinct_returns, init_returns = zip(*all_returns)


    # plot stuff
    plt.title("Performance on test set of 1000 unseen tracks")
    plot_returns(fourier_returns, "Fourier Controller")
    plot_returns(init_returns, "Init Controller")
    plot_returns(instinct_returns, "Instinct Controller")
    plt.legend()
    plt.show()
