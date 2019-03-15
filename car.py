from math import cos, sin, pi, exp
from track import Line, LapData

class Car:
    def __init__(self, x = 0.0, y = 0.0, track = None):
        self.x = x
        self.y = y
        self.dir = 0.0
        self.speed = 0.0
        
        self.prevX = x
        self.prevY = y
        
        self.steering = 0.0
        self.throttle = 0.0
        self.brake    = 0.0
        
        self.throttleCoeff = 0.05
        self.brakeCoeff = 0.0375
        self.steeringCoeff = 0.05
        self.steeringSpeedLoss = 1.0/16
        
        self.track = track
        self.lapData = None
        
        self.offRoad = False
    
    def initLapData(self):
        #if self.lapData is None:
        self.lapData = LapData(self)
    
    def update(self):

        self.enforceInputBoundaries()
        
        self.prevX = self.x
        self.prevY = self.y
        self.x += cos(self.dir)*self.speed*2 # the *2 came from an old error that I tuned the car variables around so it feels better this way...
        self.y += sin(self.dir)*self.speed*2
        
        self.speed += exp(-self.speed * 0.85) * self.throttle * self.throttleCoeff
        
        if self.track is not None:
            mLine = self.getMotionLine()
            for l in self.track.trackLines:
                if mLine.intersect(l):
                    self.offRoad = not self.offRoad
        
        self.speed -= 0.01 if self.offRoad else 0.001 # rolling friction (default was 0.001)
        brakeReduce = self.brake*self.brakeCoeff
        if (self.speed > brakeReduce):
            self.speed -= brakeReduce
        else:
            self.speed = 0
        
        
        steeringTraction = 1.0
        if (abs(self.steering) > self.getSteeringThreshold()):
            steeringTraction = 0.01

        degreesToChange = self.speed * self.steering * self.steeringCoeff * steeringTraction
        self.dir += degreesToChange
        self.speed -= abs(degreesToChange * self.steeringSpeedLoss)
        
        if self.lapData is not None: self.lapData.update()
        
    def getSteeringThreshold(self):
        return 1.125 - 0.2 * abs(self.speed + self.throttle/2.5 - self.brake/3)
        #return 3.0* self.speed / (1.0 + abs(self.throttle+self.brake)) # maybe try e^-x stuff
        
    def getMotionLine(self):
        return Line((self.prevX, self.prevY), (self.x, self.y))
    
    def enforceInputBoundaries(self):
        def clamp(my_value, min_value, max_value):
            return max(min(my_value, max_value), min_value)
        self.throttle = clamp(self.throttle, 0.0, 1.0)
        self.brake = clamp(self.brake, 0.0, 1.0)
        self.steering = clamp(self.steering, -1.0, 1.0)
    
    # def displayFunc(self):
    #     glBegin(GL_LINE_STRIP)
    #     glColor3f(0,0,0)
    #     length = 10.0
    #     angle  = pi/6.0
    #     glVertex2f(self.x + length*cos(self.dir - angle), self.y + length*sin(self.dir - angle))
    #     glVertex2f(self.x + length*cos(self.dir + angle), self.y + length*sin(self.dir + angle))
    #     glVertex2f(self.x + length*cos(self.dir + pi - angle), self.y + length*sin(self.dir + pi - angle))
    #     glVertex2f(self.x + length*cos(self.dir + pi + angle), self.y + length*sin(self.dir + pi + angle))
    #     glVertex2f(self.x + length*cos(self.dir - angle), self.y + length*sin(self.dir - angle))
    #     glEnd()