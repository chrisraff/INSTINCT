from random import seed, randint
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from fourierBasisController import *
from straightController import *
from instinctController import *


fig, ax = plt.subplots()

car_line, = plt.plot([0], [0], color='red', animated=True)

track_line_0, = plt.plot([0], [0], color='black', animated=True)
track_line_1, = plt.plot([0], [0], color='black', animated=True)

# track = pickle.load(open('default track.pickle', 'rb'))
# track = pickle.load(open('tracks/track00000.pickle', 'rb'))
# track = pickle.load(open('tracks/track00000_flip.pickle', 'rb'))
# track = pickle.load(open('tracks/track00002.pickle', 'rb'))
# track.updateTrackLines()

seed(time())
trackname = 'tracks/track{:05d}{}.pickle'.format(randint(0,5000), '' if randint(0,1) == 0 else '_flip')
print("loading track {}".format(trackname))
track = pickle.load(open(trackname, 'rb'))
track.updateTrackLines()


# print("loading population")
# pop_object = pickle.load(open('population.pickle', 'rb'))
# controller = pop_object.get_champion()
print("loading champion")
controller = pickle.load(open('champion.pickle', 'rb'))
# controller = pickle.load(open('awesome_champion.pickle', 'rb'))

# controller = StraightController(track)
# controller = pickle.load(open('instinctController.pickle', 'rb'))
# controller = pickle.load(open('default fourierBasisController.pickle', 'rb'))
# controller = pickle.load(open('fourierController.pickle', 'rb'))
controller.epsilon_min = 0
controller.epsilon = 0
controller.train = False
controller.update_track(track)

controllers = [ controller ]

controllers[0].car.initLapData()


def init():
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    return [track_line_0, track_line_1, car_line]


def update(frame):

    for controller in controllers:
        controller.update()

    focus_car = controllers[0].car

    angles = focus_car.dir + np.array( [ - np.pi/6, np.pi/6, np.pi - np.pi/6, np.pi + np.pi/6, - np.pi/6 ] )

    car_line_x = 10 * np.cos(angles)
    car_line_y = 10 * np.sin(angles)

    track_line_0_x = [ track.loop[0][i][0] - focus_car.x for i in range(len(track.loop[0])) ]
    track_line_0_y = [ track.loop[0][i][1] - focus_car.y for i in range(len(track.loop[0])) ]
    track_line_0_x += [ track.loop[0][0][0] - focus_car.x ]
    track_line_0_y += [ track.loop[0][0][1] - focus_car.y ]

    track_line_1_x = [ track.loop[1][i][0] - focus_car.x for i in range(len(track.loop[1])) ]
    track_line_1_y = [ track.loop[1][i][1] - focus_car.y for i in range(len(track.loop[1])) ]
    track_line_1_x += [ track.loop[1][0][0] - focus_car.x ]
    track_line_1_y += [ track.loop[1][0][1] - focus_car.y ]

    car_line.set_data(car_line_x, car_line_y)
    track_line_0.set_data(track_line_0_x, track_line_0_y)
    track_line_1.set_data(track_line_1_x, track_line_1_y)

    return [track_line_0, track_line_1, car_line]


print("loading visualizer")
ftime = 1000/60 # delay between frames in milliseconds

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, interval=ftime, blit=True)
plt.show()
