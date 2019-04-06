import random
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.interpolate import interp1d
from scipy import interpolate
from numpy.linalg import norm


def createCircle():
    n = 10 # number of points
    points = np.zeros((n,2))
    # theta = np.linspace(2*pi/n, 2*pi, n)
    theta = np.linspace(0, 2*pi, num=n, endpoint=False)

    for i in range(n):
        points[i,:] = [cos(theta[i]), sin(theta[i])]

    return points


def plotPoints(points):
    plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)

    for i in range(len(points)):
        if i < len(points)-1:
            plt.plot((points[i][0], points[i+1][0]), (points[i][1], points[i+1][1]), 'bo-' )
        else:
            plt.plot((points[i][0], points[0][0]), (points[i][1], points[0][1]), 'bo-' )

        # plt.pause(0.1)
    # plt.show()
    return


def alterPoints(points):
    control_points = np.copy(points)

    # push the points in and out away from the center of the circle
    min_expand_distance = 0.2
    max_expand_distance = 0.5
    max_contract_distance = 0.2

    theta = np.arctan2(points[:,1], points[:,0])
    # delta_distance = np.random.uniform(-max_contract_distance, max_expand_distance, len(points))

    # alternate pushing and pulling to increase number of turns
    delta_distance = np.empty((len(points),))
    delta_distance[::2] = 1
    delta_distance[1::2] = -1
    delta_distance *= np.random.uniform(min_expand_distance, max_expand_distance, len(points))

    noise = np.array([delta_distance * np.cos(theta), delta_distance * np.sin(theta)]).T
    control_points += noise


    # on top of the previous transformation, add a random delta
    max_uniform_delta = 0.4

    noise = np.random.uniform(-max_uniform_delta, max_uniform_delta, len(points)*2).reshape(points.shape)
    control_points = control_points + noise
    control_points[-1,:] = control_points[0,:]  # not sure why this line is needed

    # print(points)
    # print(points + noise)

    return control_points


def make_track():
    num_output_points = 1000
    track_width = 0.05


    points = createCircle()
    # plotPoints(points)
    control_points = alterPoints(points)
    # plotPoints(control_points)

    # https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
    tck, _ = interpolate.splprep(control_points.T, s=0, per=True)
    track_points = interpolate.splev(np.linspace(0, 1, num_output_points), tck)
    track_points = np.array(track_points).T

    # get the left and right sides of the track
    left_track = []
    right_track = []

    for i in range(len(track_points)):
        p0 = track_points[i-1]
        p1 = track_points[i]
        v = p1-p0  # forward

        delta_left = np.array([-v[1], v[0]])
        delta_left = track_width * delta_left / norm(delta_left)

        delta_right = np.array([v[1], -v[0]])
        delta_right = track_width * delta_right / norm(delta_right)

        left_track += [delta_left]
        right_track += [delta_right]

    left_track = np.array(left_track) + track_points
    right_track = np.array(right_track) + track_points


    left_track[0] = left_track[-1]
    right_track[0] = right_track[-1]

    return control_points, track_points, left_track, right_track


def main():
    # # plot one track for debugging
    # control_points, track_points, left_track, right_track = make_track()
    # plt.plot(left_track[:,0], left_track[:,1])
    # plt.plot(right_track[:,0], right_track[:,1])
    # # plt.plot(track_points[:,0], track_points[:,1])
    # plt.axis('equal')
    # plt.show()

    # plot lots of tracks to get a better idea of the results
    f, axes = plt.subplots(3, 5)
    # f, axes = plt.subplots(4, 8)
    f.subplots_adjust(left=0,right=1,bottom=0,top=1)
    for ax_row in axes:
        for ax in ax_row:
            control_points, track_points, left_track, right_track = make_track()
            ax.scatter(control_points[:,0], control_points[:,1], c='r')

            ax.plot(left_track[:,0], left_track[:,1], c='g')
            ax.plot(right_track[:,0], right_track[:,1], c='g')
            # ax.plot(track_points[:,0], track_points[:,1], c='b')

            ax.axis('equal')  # preserve aspect ratio
            ax.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
