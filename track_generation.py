import random
import matplotlib.pyplot as plt
import numpy as np
from math import *
# from scipy.interpolate import interp1d
from scipy import interpolate
from numpy.linalg import norm

# np.random.seed(3)

def createCircle():
    n = 12 # number of points

    #note that we need to make an extra point because the last one has to be set to the same as the first
    points = np.zeros((n+1,2))
    theta = np.linspace(0, 2*pi, num=n+1, endpoint=True)

    for i in range(n+1):
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
    theta = np.arctan2(points[:,1], points[:,0])

    max_uniform_delta_radius = 0.4
    max_contract_distance = -0.2
    max_expand_distance = 0.5

    # push the points in and out away from the center of the circle
    # delta_distance = np.random.uniform(-max_contract_distance, max_expand_distance, len(points))
    # noise = np.array([delta_distance * np.cos(theta), delta_distance * np.sin(theta)]).T
    # control_points += noise

    # alternate pushing and pulling to increase number of turns
    delta_distance = np.empty((len(points),))
    delta_distance[::2] = 1
    delta_distance[1::2] = -1
    delta_distance *= np.random.uniform(max_contract_distance, max_expand_distance, len(points))
    noise = np.array([delta_distance * np.cos(theta), delta_distance * np.sin(theta)]).T
    control_points += noise

    # add a random delta to each control point, sampled uniformly over a circle
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
    dr = max_uniform_delta_radius * np.sqrt(np.random.uniform(size=len(points)))
    dt = np.random.uniform(size=len(points)) * 2 * pi
    control_points[:,0] += dr*np.cos(dt)
    control_points[:,1] += dr*np.sin(dt)

    control_points[-1,:] = control_points[0,:]  # this is required by the interpolation functions
    # print(control_points)
    # print(control_points + noise)

    return control_points


def make_track():
    num_output_points = 50
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


def find_intersections(control_points, track_points, left_track, right_track):

    # https://stackoverflow.com/a/9997374/2230446
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # Return true if line segments AB and CD intersect
    def lines_intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    from tqdm import trange
    for i in range(len(left_track)):
        A = left_track[i-1]
        B = left_track[i]

        # may need to optimize length of this inner loop if the generator is too slow
        for j in range(len(right_track)):
            C = right_track[j-1]
            D = right_track[j]

            has_intersection = lines_intersect(A, B, C, D)
            if has_intersection and i != 1:
                # print("found left-right intersection at i={}".format(i))
                # plt.plot(A[0], A[1], 'bo-', alpha=0.5)
                # plt.plot(B[0], B[1], 'bo-', alpha=0.5)
                # plt.plot(C[0], C[1], 'yo-', alpha=0.5)
                # plt.plot(D[0], D[1], 'yo-', alpha=0.5)
                return True

        # may need to optimize length of this inner loop if the generator is too slow
        for j in range(i-len(left_track)//2,i-2):
            C = left_track[j-1]
            D = left_track[j]

            has_intersection = lines_intersect(A, B, C, D)
            if has_intersection:
                # print("found left-left intersection at i={}".format(i))
                # plt.plot(A[0], A[1], 'bo-', alpha=0.5)
                # plt.plot(B[0], B[1], 'bo-', alpha=0.5)
                # plt.plot(C[0], C[1], 'yo-', alpha=0.5)
                # plt.plot(D[0], D[1], 'yo-', alpha=0.5)
                return True

    return False


def main():
    # # plot one track for debugging
    # control_points, track_points, left_track, right_track = make_track()
    # has_intersection = find_intersections(control_points, track_points, left_track, right_track)
    # print("has_intersection: {}".format(has_intersection))
    # plt.plot(left_track[:,0], left_track[:,1])
    # plt.plot(right_track[:,0], right_track[:,1])
    # plt.scatter(control_points[:,0], control_points[:,1], color='r')
    # # plt.plot(track_points[:,0], track_points[:,1])
    # plt.axis('equal')
    # plt.show()

    # plot lots of tracks to get a better idea of the results
    f, axes = plt.subplots(5, 8)
    # f, axes = plt.subplots(4, 8)
    f.subplots_adjust(left=0,right=1,bottom=0,top=1)
    track_num = 0
    for ax_row in axes:
        for ax in ax_row:
            track_num += 1
            print("generating track {} / {}".format(track_num, len(axes)*len(axes[0])))
            track_data = make_track()
            while find_intersections(*track_data):
                print("  rejecting track")
                track_data = make_track()
            control_points, track_points, left_track, right_track = track_data
            ax.scatter(control_points[:,0], control_points[:,1], c='r')

            ax.plot(left_track[:,0], left_track[:,1], c='g')
            ax.plot(right_track[:,0], right_track[:,1], c='g')
            # ax.plot(track_points[:,0], track_points[:,1], c='b')

            ax.axis('equal')  # preserve aspect ratio
            ax.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
