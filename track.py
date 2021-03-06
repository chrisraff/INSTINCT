import pickle

class Track:
    def __init__(self):
        self.loop = [[],[]] # two arrays of points
        self.trackLines = [] # array of lines that correspond with loops for intersection checking
        self.checkpoints = [] # array of lines
        self.start_position = (0,0) # point
        self.start_direction = 0 # radians
        
        # self.showCheckpoints = True
        

    def updateTrackLines(self):
        self.trackLines = []
        for l in self.loop:
            for n in range(len(l)):
                self.trackLines+=[Line(l[n], l[n-1])]
    
    # def displayFunc(self):
    #     glColor3f(0,0,0)
    #     for l in self.loop:
    #         glBegin(GL_LINE_LOOP)
    #         for p in l:
    #             glVertex2f(p[0], p[1])
    #         glEnd()
    #     glBegin(GL_LINES)
    #     # sectors
    #     glColor3f(0.8,0,0.8)
    #     for l in self.sectors:
    #         glVertex2f(l.p[0][0], l.p[0][1])
    #         glVertex2f(l.p[1][0], l.p[1][1])
    #     # start/finish line
    #     glColor3f(0,1,0)
    #     glVertex2f(self.start.p[0][0], self.start.p[0][1])
    #     glVertex2f(self.start.p[1][0], self.start.p[1][1])
        
    #     glEnd()
    #     self.showCheckpoints = True
        
    #     if self.showCheckpoints:
    #         from math import sin, cos, pi
    #         from numpy import linspace
    #         glColor3f(0.8, 0.8, 0)
    #         for c in self.checkpoints:
    #             glBegin(GL_LINE_LOOP)
    #             for n in linspace(0, 2*pi, 24, endpoint = False):
    #                 glVertex2f(c[0]+c[2]*cos(n), c[1]+c[2]*sin(n))
    #             glEnd()
        
class Line:
    def __init__(self, p0, p1):
        self.p = [p0, p1]
        

    # Return true if line segments l0 and l1 intersect (ignores colinearity)
    def intersect(self, l1):
        def ccw(A,B,C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(self.p[0],l1.p[0],l1.p[1]) != ccw(self.p[1],l1.p[0],l1.p[1]) and ccw(self.p[0],self.p[1],l1.p[0]) != ccw(self.p[0],self.p[1],l1.p[1])

    def point_dist(self, point):
        px = self.p[1][0] - self.p[0][0]
        py = self.p[1][1] - self.p[0][1]

        some_norm = px*px + py*py

        u =  ((point[0] - self.p[0][0]) * px + (point[1] - self.p[0][1]) * py) / float(some_norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = self.p[0][0] + u * px
        y = self.p[0][1] + u * py

        dx = x - point[0]
        dy = y - point[1]

        # Note: If the actual distance does not matter,
        # if you only want to compare what this function
        # returns to other results of this function, you
        # can just return the squared distance instead
        # (i.e. remove the sqrt) to gain a little performance

        dist = (dx*dx + dy*dy)**.5

        return dist

   
class LapData:

    # car: The car that the lapData will record data for
    def __init__(self, car):
        self.laps = [] # array of times
        self.track = car.track
        self.nextCheckpoint = 0
        self.numCheckpoints = len(self.track.checkpoints)
        self.checkpointDists = self.initCheckpointDists()
        self.checkpointNetDists = [0] + [sum(self.checkpointDists[:n+1]) for n in range(self.numCheckpoints)] # could just use cumulative sum...
        self.car = car
        self.startTime = 0 # the current lap's start time
        self.time = 0


    def initCheckpointDists(self):
        ret = []
        for n in range(self.numCheckpoints):
            c0 = self.track.checkpoints[n]
            c1 = self.track.checkpoints[(n+1)%self.numCheckpoints]
            c0_average = [ (c0.p[0][0] + c0.p[1][0]) / 2, (c0.p[0][1] + c0.p[1][1]) / 2]
            c1_average = [ (c1.p[0][0] + c1.p[1][0]) / 2, (c1.p[0][1] + c1.p[1][1]) / 2]
            dist = ((c0_average[0]-c1_average[0])**2 + (c0_average[1]-c1_average[1])**2)**0.5
            ret += [dist]
        return ret
        

    # returns True if the lap was valid (unused)
    def isNewLap(self):
    # if all checkpoints hit (lap is valid)
        valid = self.nextCheckpoint == self.numCheckpoints
        if valid:
            self.laps += [self.time - self.startTime]
            print(self.laps)
        self.startTime = self.time
        self.nextCheckpoint = 0

        return valid


    # returns the number of checkpoints crossed (this could be changed to fractional lap progress made)
    def update(self):
        self.time += 1/60.0

        # see if we passed any checkpoints
        carMotionLine = self.car.getMotionLine()

        result = 0
        while self.track.checkpoints [self.nextCheckpoint] .intersect( carMotionLine ):
            result += 1
            # print('checkpoint {}'.format(self.nextCheckpoint))

            self.nextCheckpoint = (self.nextCheckpoint + 1) % self.numCheckpoints

        return result
            
            
    def getDistanceToNextCheckpoint(self):
        return self.getDistanceToCheckpoint(self.nextCheckpoint)


    def getDistanceToCheckpoint(self, checkpoint_index):
        return self.track.checkpoints[checkpoint_index] .point_dist( (self.car.x, self.car.y) )

        
    def getProgress(self):
        distance = self.checkpointNetDists[self.nextCheckpoint]
        
        dist_to_next = self.track.checkpoints[self.nextCheckpoint] .point_dist ( (self.car.x, self.car.y) )
        dist_to_last = self.track.checkpoints[ (self.nextCheckpoint - 1) % self.numCheckpoints ] .point_dist ( (self.car.x, self.car.y) )

        normed_section_dist = dist_to_last / (dist_to_next + dist_to_last)

        distance += normed_section_dist * self.checkpointDists[self.nextCheckpoint]

        return distance / self.checkpointNetDists[-1]
        
    
        
class Editor:
    def __init__(self, track = Track()):
        self.mode = 0
        self.append = 0
        self.cursor = (0, 0)
        self.track = track
        self.checkpointClickState = True
        

    def keyboardFunc(self, key, x, y):
        # consider putting some lock that prevents changing the mode until 'edit mode' has been enabled
        if key in '0123456789' and self.checkpointClickState:
            self.mode = int(key)
            
        # s saves
        elif key == 's':
            pickle.dump(self.track, open('track', 'w'))
        # l loads
    

    def passiveMotionFunc(self, x, y): # currently unused
        if (self.mode == 3 and not self.checkpointClickState):
            cid = len(self.track.checkpoints)-1
            cx = self.track.checkpoints[cid][0]
            cy = self.track.checkpoints[cid][1]
            r = ((cx - x)**2 + (cy - y)**2)**0.5
            self.track.checkpoints[cid] = (cx, cy, r)
            
        
    # def mouseFunc(self, button, state, x, y):
    #     if self.mode in [1, 2]:
    #         if button == GLUT_LEFT_BUTTON and state == GLUT_UP: # Hopefully, this is on release
    #             if self.append >= len(self.track.loop[self.mode-1]):
    #                 self.track.loop[self.mode-1] += [(x, y)]
    #             elif self.append >= 0:
    #                 self.track.loop[self.mode-1] = self.track.loop[self.mode-1][:self.append] + [(x,y)] + self.track.loop[self.mode-1][self.append+1:]
    #             #self.track.updateTrackLines()
    #             self.append += 1
    #     #3 add checkpoints
    #     if self.mode == 3:
    #         if button == GLUT_LEFT_BUTTON and state == GLUT_UP:
    #             if self.checkpointClickState:
    #                 self.track.checkpoints += [(x, y, 0.0)]
                    
    #                 self.checkpointClickState = False
    #             else:
    #                 cid = len(self.track.checkpoints)-1
    #                 cx = self.track.checkpoints[cid][0]
    #                 cy = self.track.checkpoints[cid][1]
    #                 r = ((cx - x)**2 + (cy - y)**2)**0.5
    #                 self.track.checkpoints[cid] = (cx, cy, r)
                    
    #                 self.checkpointClickState = True
    #     # 4 add sectors
    #     # 5 move points?
        
    # def displayFunc(self):
    #     # display the checkpoints
    #     glColor3f(0.8, 0.8, 0)
    #     from math import sin, cos, pi
    #     from numpy import linspace
    #     for c in self.track.checkpoints:
    #         glBegin(GL_LINE_LOOP)
    #         for n in linspace(0, 2*pi, 24, endpoint = False):
    #             glVertex2f(c[0]+c[2]*cos(n), c[1]+c[2]*sin(n))
    #         glEnd()