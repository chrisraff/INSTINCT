import car
import numpy as np
from math import exp
from controllerUtils import getDistanceReadings


class linearBiasController:
    def __init__(self, track = None, steeringWeights = None, thrustWeights = None, steeringBias = 0.0, thrustBias = 0.0):
        self.car = car.Car(track.start_position[0], track.start_position[1], track)
        self.car.dir = track.start_direction
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

        self.percepts[:self.numDistSensors] = getDistanceReadings(self.car, self.track, self.numDistSensors)
        
        self.percepts[-1] = self.car.speed
        
        steering = np.dot(self.steeringWeights, self.percepts) + self.steeringBias
        thrust   = np.dot(self.thrustWeights, self.percepts) + self.thrustBias
        self.car.steering =  steering
        self.car.throttle =  thrust if thrust > 0 else 0
        self.car.brake    = -thrust if thrust < 0 else 0
        
        self.car.update()
        
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


