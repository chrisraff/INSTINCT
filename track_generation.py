from multiprocessing import Pool, cpu_count
import pickle
from tqdm import trange, tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy import interpolate
from numpy.linalg import norm
from track import Track, Line


np.random.seed(6)

# track settings
num_tracks_to_generate = 10
num_control_points = 12
num_track_points = 100
track_width = 0.1

# randomization settings
max_uniform_delta_radius = 0.4
max_contract_distance = -0.2
max_expand_distance = 0.5


# get perpendicular vector to input vector a
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def create_circle():
    #note that we need to make an extra point because the last one has to be set to the same as the first
    points = np.zeros((num_control_points+1,2))
    theta = np.linspace(0, 2*pi, num=num_control_points+1, endpoint=True)

    for i in range(num_control_points+1):
        points[i,:] = [cos(theta[i]), sin(theta[i])]

    return points


def plot_points(points):
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


def alter_points(points):
    control_points = np.copy(points)
    theta = np.arctan2(points[:,1], points[:,0])

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
    points = create_circle()
    # plot_points(points)
    control_points = alter_points(points)
    # plot_points(control_points)

    # https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
    tck, _ = interpolate.splprep(control_points.T, s=0, per=True)
    track_points = interpolate.splev(np.linspace(0, 1, num_track_points), tck)
    track_points = np.array(track_points).T

    # get the left and right sides of the track
    left_track = []
    right_track = []

    for i in range(len(track_points)):
        p0 = track_points[i-2]
        p1 = track_points[i-1]
        p2 = track_points[i]
        # v = p2-p1  # forward
        v = ((p1-p0)+(p2-p1))/2  # vector averaging for more smoothness

        if i == 0:
            left_track += [np.array([0, 0])]
            right_track += [np.array([0, 0])]
            continue

        delta_left = perp(v)
        delta_left = track_width/2 * delta_left / norm(delta_left)

        delta_right = -perp(v)
        delta_right = track_width/2 * delta_right / norm(delta_right)

        left_track += [delta_left]
        right_track += [delta_right]

    left_track = np.array(left_track) + track_points
    right_track = np.array(right_track) + track_points


    left_track[0] = left_track[-1]
    right_track[0] = right_track[-1]
    # left_track[-1] = left_track[0]
    # right_track[-1] = right_track[0]

    return control_points, track_points, left_track, right_track


def nudge_points(control_points, track_points, left_track, right_track):

    # returns the distance from point p to the line segment denoted by the two points l0 and l1
    # https://stackoverflow.com/a/39840218/2230446
    # def point_line_dist(p, line_start, line_end):
    #     return norm(np.cross(line_end-line_start, line_start-p))/norm(line_end-line_start)
    # def point_line_dist(p1, p2, p3):
    #     # return np.abs(norm(np.cross(p2-p1, p1-p3))/norm(p2-p1))
    #     return np.abs(np.cross(p2-p1, p1-p3)) / norm(p2-p1)

    # https://stackoverflow.com/a/2233538/2230446
    def point_line_dist(x1, y1, x2, y2, x3, y3): # x3,y3 is the point
        px = x2-x1
        py = y2-y1

        some_norm = px*px + py*py

        u =  ((x3 - x1) * px + (y3 - y1) * py) / float(some_norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        # Note: If the actual distance does not matter,
        # if you only want to compare what this function
        # returns to other results of this function, you
        # can just return the squared distance instead
        # (i.e. remove the sqrt) to gain a little performance

        dist = (dx*dx + dy*dy)**.5

        return dist


    def nudge_tracks(track1, track2):
        for i in range(len(track1)):
            A = track1[i]

            min_dist = 1e9
            min_B = None
            min_C = None

            for j in range(len(track2)):
                B = track2[j-1]
                C = track2[j]

                # "2*" is for DEBUGGING ONLY
                # dist_to_track = point_line_dist(B, C, A)

                if np.allclose(B, C):
                    dist_to_track = norm(B-A)
                else:
                    dist_to_track = point_line_dist(*B, *C, *A)

                if dist_to_track < min_dist:
                    min_dist = dist_to_track
                    min_B = B
                    min_C = C

                # if dist_to_track < track_width-1e-5:
                #     # points_are_too_close = True
                #     # print("NUDGING i={} j={}".format(i, j))
                #     nudge_direction = perp(C-B)
                #     nudge_direction = nudge_direction/norm(nudge_direction)
                #     # print(nudge_direction)
                #     plt.scatter([B[0]], [B[1]], color='red', alpha=0.2)
                #     plt.scatter([C[0]], [C[1]], color='red', alpha=0.2)
                #     plt.scatter([A[0]], [A[1]], c='black', alpha=0.2)

                #     # print("nudge amount: {}".format(track_width-dist_to_track))

                #     track1[i] += 0.5*(track_width-dist_to_track)*nudge_direction
                #     pass

            # print("min_dist: {}".format(min_dist))
            if min_dist < track_width:
                # points_are_too_close = True
                # print("NUDGING i={} j={}".format(i, j))
                nudge_direction = perp(min_C-min_B)
                nudge_direction = nudge_direction/norm(nudge_direction)
                # print(nudge_direction)
                # plt.scatter([B[0]], [B[1]], color='red', alpha=0.2)
                # plt.scatter([C[0]], [C[1]], color='red', alpha=0.2)
                # plt.scatter([A[0]], [A[1]], c='black', alpha=0.2)

                # print("nudge amount: {}".format(track_width-min_dist))

                track1[i] += 1.0*(track_width-min_dist)*nudge_direction
                pass

    nudge_tracks(left_track, right_track)
    # nudge_tracks(right_track, left_track)

    pass


def find_intersections(control_points, track_points, left_track, right_track):
    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    # return
    def seg_intersect(a1,a2, b1,b2) :
        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = perp(da)
        denom = np.dot( dap, db)
        num = np.dot( dap, dp )
        return (num / denom.astype(float))*db + b1

    # https://stackoverflow.com/a/9997374/2230446
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # Return true if line segments AB and CD intersect
    def lines_intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    # check for intersections between the left and right tracks
    for i in range(len(left_track)):
        A = left_track[i-1]
        B = left_track[i]

        # may need to optimize length of this inner loop if the generator is too slow
        for j in range(len(right_track)):
            C = right_track[j-1]
            D = right_track[j]

            has_intersection = lines_intersect(A, B, C, D)
            if has_intersection and i != 1:
                # print("found left-right intersection at i={} j={}".format(i, j))
                # plt.plot(A[0], A[1], 'bo-', alpha=0.5)
                # plt.plot(B[0], B[1], 'bo-', alpha=0.5)
                # plt.plot(C[0], C[1], 'yo-', alpha=0.5)
                # plt.plot(D[0], D[1], 'yo-', alpha=0.5)
                return True

    def correct_self_intersection(side_track):
        # may need to optimize length of this inner loop if the generator is too slow
        for i in range(len(side_track)):
            A = side_track[i-1]
            B = side_track[i]

            if np.allclose(A, B):
                # print("skipping AB vectors that are touching")
                continue

            for j in range(i-len(side_track)//2,i-2):
                C = side_track[j-1]
                D = side_track[j]

                if np.allclose(A, D):
                    # print("skipping AD vectors that are touching")
                    continue
                if np.allclose(B, C):
                    # print("skipping BC vectors that are touching")
                    continue

                has_intersection = lines_intersect(A, B, C, D)
                if has_intersection:
                    intersection_point = seg_intersect(A, B, C, D)
                    # print("found left-left intersection at i={} j={} at location {}".format(i, j, intersection_point))
                    # plt.scatter([intersection_point[0]], [intersection_point[1]], color='k')
                    # print("ABCD points are {} {} {} {}".format(A, B, C, D))
                    # plt.plot(A[0], A[1], 'bo-', alpha=0.5)
                    # plt.plot(B[0], B[1], 'bo-', alpha=0.5)
                    # plt.plot(C[0], C[1], 'yo-', alpha=0.5)
                    # plt.plot(D[0], D[1], 'yo-', alpha=0.5)

                    # set all the points in the loop to the intersection point
                    smaller_index, bigger_index = min(i, j-1), max(i, j-1)
                    if smaller_index < 0:
                        side_track[smaller_index:] = intersection_point
                        side_track[:bigger_index] = intersection_point
                    else:
                        side_track[smaller_index:bigger_index] = intersection_point

                    return True
        return False

    there_was_a_self_intersection = True
    while(there_was_a_self_intersection):
        there_was_a_self_intersection = correct_self_intersection(left_track) or correct_self_intersection(right_track)

    return there_was_a_self_intersection


def scale_track(control_points, track_points, left_track, right_track):
    goal_track_width = 96
    scaling_factor = goal_track_width / track_width
    control_points *= scaling_factor
    track_points *= scaling_factor
    left_track *= scaling_factor
    right_track *= scaling_factor



def track_to_track_object(control_points, track_points, left_track, right_track):

    scale_track(control_points, track_points, left_track, right_track)

    some_track = Track()

    for i in range(len(left_track)-1):
        # using line checkpoints
        some_track.checkpoints += [ Line((left_track[i][0], left_track[i][1]), (right_track[i][0], right_track[i][1])) ]

    # remove the overlapping points as per chris' recommendation
    def without_duplicate_vectors(a):
        _, idx = np.unique(a.round(decimals=6), return_index=True, axis=0)
        return a[np.sort(idx)]

    left_track = without_duplicate_vectors(left_track)
    right_track = without_duplicate_vectors(right_track)
    track_points = without_duplicate_vectors(track_points)
    control_points = without_duplicate_vectors(control_points)

    some_track.loop[0] = [(x[0], x[1]) for x in left_track]
    some_track.loop[1] = [(x[0], x[1]) for x in right_track]

    start_direction_vector = track_points[0]-track_points[-1]
    start_point = (left_track[0]+left_track[-1]+right_track[0]+right_track[-1])/4
    some_track.start_position = start_point
    some_track.start_direction = np.arctan2(start_direction_vector[1], start_direction_vector[0])

    # plt.plot(left_track[:,0], left_track[:,1], c='r')
    # plt.plot(right_track[:,0], right_track[:,1], c='g')
    # plt.scatter([start_point[0]], [start_point[1]], color='k')
    # plt.scatter([start_point[0]+start_direction_vector[0]], [start_point[1]+start_direction_vector[1]], color='b')
    # plt.scatter([0], [0], color='cyan')
    # plt.axis('equal')
    # plt.show()

    return some_track


def make_track_object():
    track_data = make_track()
    while find_intersections(*track_data):
        # print("  rejecting track")
        track_data = make_track()
    nudge_points(*track_data)
    track_object = track_to_track_object(*track_data)
    return track_object


def multiprocessing_generate_track(track_num):
    np.random.seed(track_num)
    some_track = make_track_object()
    pickle.dump( some_track, open( 'tracks/track{:05d}.pickle'.format(track_num), 'wb' ) )


def main():
    # # plot one track for debugging
    # control_points, track_points, left_track, right_track = make_track()
    # has_intersection = find_intersections(control_points, track_points, left_track, right_track)
    # # print("has_intersection: {}".format(has_intersection))
    # nudge_points(control_points, track_points, left_track, right_track)
    # plt.plot(left_track[:,0], left_track[:,1], c='r')
    # plt.plot(right_track[:,0], right_track[:,1], c='g')
    # plt.scatter(control_points[:,0], control_points[:,1], color='b', alpha=0.5)
    # plt.plot(track_points[:,0], track_points[:,1], c='b')
    # plt.axis('equal')
    # plt.show()


    # save tracks to pickle files
    # for track_num in trange(num_tracks_to_generate):
    #     multiprocessing_generate_track(track_num)
    with Pool(cpu_count()) as p:
        r = list(tqdm(p.imap(multiprocessing_generate_track, range(num_tracks_to_generate)), total=num_tracks_to_generate))


    # # plot lots of tracks to get a better idea of the results
    # f, axes = plt.subplots(2, 4)
    # # f, axes = plt.subplots(4, 8)
    # f.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # track_num = 0
    # print("Generating tracks")
    # with tqdm(total=len(axes)*len(axes[0])) as pbar:
    #     for ax_row in axes:
    #         for ax in ax_row:
    #             track_num += 1
    #             # print("generating track {} / {}".format(track_num, len(axes)*len(axes[0])))
    #             track_data = make_track()
    #             while find_intersections(*track_data):
    #                 print("  rejecting track")
    #                 track_data = make_track()
    #             nudge_points(*track_data)
    #             control_points, track_points, left_track, right_track = track_data
    #             ax.scatter(control_points[:,0], control_points[:,1], c='b', alpha=0.5)

    #             ax.plot(left_track[:,0], left_track[:,1], c='r')
    #             ax.plot(right_track[:,0], right_track[:,1], c='g')
    #             # ax.plot(track_points[:,0], track_points[:,1], c='b')

    #             ax.axis('equal')  # preserve aspect ratio
    #             ax.axis('off')

    #             pbar.update(1)
    # plt.show()

if __name__ == "__main__":
    main()
