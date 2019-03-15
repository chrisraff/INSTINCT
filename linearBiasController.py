import car
import numpy as np
from math import sin, cos, pi, exp

class linearBiasController:
    def __init__(self, track = None, steeringWeights = None, thrustWeights = None, steeringBias = 0.0, thrustBias = 0.0):
        self.myCar = car.Car(0.0, 0.0, track)
        self.track = track
        
        # controller variables go here
        self.numDistSensors = 5
        self.steeringWeights = steeringWeights
        if steeringWeights is None:
            self.steeringWeights = [np.random.randn() for _ in range(self.numDistSensors+1)]
        self.thrustWeights = thrustWeights
        if thrustWeights is None:
            self.thrustWeights = [np.random.randn() for _ in range(self.numDistSensors+1)]
        self.percepts = [0.0 for _ in range(self.numDistSensors+1)] # 1 input for speed

        self.steeringBias = steeringBias
        self.thrustBias = thrustBias

        
    def update(self):
        # controller logic goes here
        percept = 0
        for n in np.linspace(-pi/2, pi/2, self.numDistSensors):
            nearest = float('inf')
            for l in self.track.trackLines:
                t = intersectsAt((self.myCar.x, self.myCar.y), (cos(self.myCar.dir+n), sin(self.myCar.dir+n)), l)
                if t > 0 and t < nearest:
                    nearest = t
            self.percepts[percept] = nearest
            percept += 1
        self.percepts[percept] = self.myCar.speed
        
        steering = np.dot(self.steeringWeights, self.percepts) + self.steeringBias
        thrust   = np.dot(self.thrustWeights, self.percepts) + self.thrustBias
        self.myCar.steering =  steering
        self.myCar.throttle =  thrust if thrust > 0 else 0
        self.myCar.brake    = -thrust if thrust < 0 else 0
        
        self.myCar.update()
        
    # def displayFunc(self):
    #     # display things about the controller (like the percepts)
        
    #     glBegin(GL_LINES)
    #     for n in range(self.numDistSensors): # np.linspace(-pi/2, pi/2, self.numDistSensors):
    #         '''nearest = 1000
    #         for l in self.track.trackLines:
    #             t = intersectsAt((self.myCar.x, self.myCar.y), (cos(self.myCar.dir+n), sin(self.myCar.dir+n)), l)
    #             if t > 0 and t < nearest:
    #                 nearest = t'''
    #         glColor3f(0,0,0)
    #         glVertex2f(self.myCar.x, self.myCar.y)
    #         fract = pi/(self.numDistSensors-1)
    #         glVertex2f(self.myCar.x + self.percepts[n]*cos(self.myCar.dir-pi/2+fract*n), self.myCar.y + self.percepts[n]*sin(self.myCar.dir-pi/2+fract*n))
    #     glEnd()
        
    #     self.myCar.displayFunc()
        
# takes a point (2d), a direction (2d), and a track.Line
def intersectsAt(p, d, l):
    def mag(v):
        return (v[0]**2 + v[1]**2)**0.5
    # normalize d
    ddenom = mag(d)
    d = (d[0]/ddenom, d[1]/ddenom)
    
    # get L normal
    lX = l.p[0][0] - l.p[1][0]
    lY = l.p[0][1] - l.p[1][1]
    llength = (lX**2 + lY**2)**0.5
    lN = (-lY/llength, lX/llength)
    
    # calculate t
    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1]
    disp = (l.p[0][0] - p[0], l.p[0][1] - p[1])
    denom = dot(d, lN)
    if denom == 0:
        return float('inf')
    t = dot(disp, lN)/denom
    
    # see if t is valid
    '''def proj(a, b): # project a onto b
        dist = dot(a, b)/(b[0]**2 + b[1]**2)
        return (b[0]*dist, b[1]*dist)'''
    def comp(a, b): # project a onto b (assumes b has unit length)
        return dot(a, b)/(b[0]**2 + b[1]**2)
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