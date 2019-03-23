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