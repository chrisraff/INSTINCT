import random
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.interpolate import interp1d
from scipy import interpolate


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
    newpoints = np.copy(points)

    # push the points in and out away from the center of the circle
    max_expand_distance = 0.5
    max_contract_distance = 0.2

    theta = np.arctan2(points[:,1], points[:,0])
    delta_distance = np.random.uniform(-max_contract_distance, max_expand_distance, len(points))
    noise = np.array([delta_distance * np.cos(theta), delta_distance * np.sin(theta)]).T
    newpoints += noise

    # on top of the previous transformation, add a random delta
    max_uniform_delta = 0.5

    noise = np.random.uniform(-max_uniform_delta, max_uniform_delta, len(points)*2).reshape(points.shape)
    newpoints = newpoints + noise
    newpoints[-1,:] = newpoints[0,:]  # not sure why this line is needed

    # print(points)
    # print(points + noise)

    return newpoints


def main():
    points = createCircle()
    # plotPoints(points)
    newpoints = alterPoints(points)
    plotPoints(newpoints)

    num_output_points = 1000

    # https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
    tck, _ = interpolate.splprep(newpoints.T, s=0, per=True)
    out = interpolate.splev(np.linspace(0, 1, num_output_points), tck)

    plt.plot(out[0], out[1])
    plt.show()


if __name__ == "__main__":
    main()
