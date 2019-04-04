import random
import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.interpolate import interp1d
from scipy import interpolate


def createCircle():
    n = 10 # number of points
    points = np.zeros((n,2))
    # theta = np.linspace(2*m.pi/n, 2*m.pi, n)
    theta = np.linspace(0, 2*m.pi, n)

    for i in range(n):
        points[i,:] = [m.cos(theta[i]), m.sin(theta[i])]

    return points

def plotPoints(points):
    plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    for i in range(len(points)):
        if i < len(points)-1:
            plt.plot((points[i][0], points[i+1][0]), (points[i][1], points[i+1][1]), 'bo-' )
        else:
            plt.plot((points[i][0], points[0][0]), (points[i][1], points[0][1]), 'bo-' )
 
        # plt.pause(0.1)
    plt.show()
    return

def alterPoints(points):
    noise = np.random.uniform(-0.5, 0.5, len(points)*2).reshape(points.shape)
    newpoints = points + noise
    newpoints[-1,:] = newpoints[0,:]
    # print(points)
    # print(points + noise)

    return newpoints




def main():
    
    points = createCircle()
    # plotPoints(points)
    newpoints = alterPoints(points)

    tck, u = interpolate.splprep([newpoints[:,0], newpoints[:,1]], s=0)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)


    plt.plot(out[0], out[1])
    plt.show()



    

    


if __name__ == "__main__":
    main()