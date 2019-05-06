import car
from track import LapData

class StraightController:

    def __init__(self, track):
        self.car = car.Car(track.start_position[0], track.start_position[1], track)
        self.car.dir = track.start_direction
        self.track = track

        self.car.initLapData()
        
        # controller variables go here
        # if there were any
        

    def update(self):
        # controller logic goes here
        self.car.throttle  = 1
        self.car.brake   = 0
        self.car.steering= 0

        self.car.update()

    
    def update_track(self, track):
        self.track = track
        
        # reset the car
        self.car = car.Car(self.track.start_position[0], self.track.start_position[1], self.track)
        self.car.dir = self.track.start_direction
        self.car.initLapData()
