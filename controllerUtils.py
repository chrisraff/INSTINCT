# this file contains general functions used by controllers to perceive the track
from numpy import linspace, sin, cos, pi

# used for loading tracks
import pickle
from os import scandir


# 
def getDistanceReadings(car, track, numDistSensors):
    percepts = [0] * numDistSensors

    for i, n in enumerate(linspace(-pi/2, pi/2, numDistSensors)):
        nearest = float('inf')
        for l in track.trackLines:
            t = intersectsAt((car.x, car.y), (cos(car.dir+n), sin(car.dir+n)), l)
            if t > 0 and t < nearest:
                nearest = t
        percepts[i] = nearest

    return percepts


# functions used in line intersection calculation
def mag(v):
    return (v[0]**2 + v[1]**2)**0.5
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]
def comp(a, b): # project a onto b (assumes b has unit length)
    return dot(a, b)/(b[0]**2 + b[1]**2)


# takes a point (2d), a direction (2d), and a track.Line
def intersectsAt(p, d, l):
    # normalize d
    ddenom = mag(d)
    d = (d[0]/ddenom, d[1]/ddenom)
    
    # get L normal
    lX = l.p[0][0] - l.p[1][0]
    lY = l.p[0][1] - l.p[1][1]
    llength = (lX**2 + lY**2)**0.5

    if llength == 0:
        # print('bad')
        return float('inf')
    lN = (-lY/llength, lX/llength)
    
    # calculate t
    disp = (l.p[0][0] - p[0], l.p[0][1] - p[1])
    denom = dot(d, lN)
    if denom == 0:
        return float('inf')
    t = dot(disp, lN)/denom
    
    # see if t is valid
    '''def proj(a, b): # project a onto b
        dist = dot(a, b)/(b[0]**2 + b[1]**2)
        return (b[0]*dist, b[1]*dist)'''
    
    '''p0p = proj(disp, d)
    p1p = proj((disp[0] - lX, disp[1] - lY), d)'''
    # p0p = mag(proj((l.p[0][0] - p[0], l.p[0][1] - p[1]), d))
    # p1p = mag(proj((l.p[1][0] - p[0], l.p[1][1] - p[1]), d))
    p0p = comp((l.p[0][0] - p[0], l.p[0][1] - p[1]), d)
    p1p = comp((l.p[1][0] - p[0], l.p[1][1] - p[1]), d)
    # print('{} - {} - {}'.format(p0p, t, p1p))
    if t >= min(p0p, p1p) and t <= max(p0p, p1p):
        return t
    else:
        return float('inf')


# Load all tracks in a specified path (like "tracks/."). The tqdm argument allows you to track the progress by passing the tqdm function
def load_tracks(path, tqdm=lambda some_list: some_list):
    def safe_load(file):
        with open(file, 'rb') as f:
            meat = pickle.load(f)
        return meat
    return [safe_load(track) for track in tqdm(scandir(path)) if track.is_file()]
