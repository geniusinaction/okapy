from okada_wrapper import dc3dwrapper
from math import sin, cos, tan, radians
import numpy as np

def rect_shear_fault(fparams, eparams, data): 
    "Attempt at a functional Okada dislocation model - wish me luck!"
    
    # fparams is a vector of fault parameter input
    #   9 numbers: strike, dip, rake, slip, x, y, length, top, bottom
    #      angles in degrees, positions/dimensions in meters
    # eparams is a vector of Lame elastic parameters
    #   2 numbers: lambda and mu (rigidity)
    #      values in N m  
    # data is an array of input x and y coordinates, displacements and LOS vectors
    #   minimum 6 numbers: x_pos, y_pos, displacement, x_los, y_los, z_los
    #      positions, displacement in meters, rest are unit los vector components 

    # fault parameters
    strike = fparams[0]
    dip = fparams[1]
    rake = fparams[2]
    slip = fparams[3]
    xs = fparams[4]
    ys = fparams[5]
    as_length = fparams[6]
    dd_width = (fparams[8]-fparams[7])/sin(radians(dip))
    cd_depth = (fparams[7]+fparams[8])/2
    
    # elastic parameters    
    lmda = eparams[0]
    mu = eparams[1]
    alpha = (lmda + mu) / (lmda + 2 * mu) # elastic constant used by Okada

    # calculate centroid coordinates
    rc = cd_depth/tan(radians(dip))  # radial surface distance from (xs,ys) to centroid
    rcx = rc*sin(radians(strike+90)) # coordinate shift in x from xs to centroid 
    rcy = rc*cos(radians(strike+90)) # coordinate shift in y from ys to centroid
    xc = xs+rcx  # x coordinate of centroid
    yc = ys+rcy  # y coordinate of centroid

    # make a rotation matrix to account for strike
    R=np.array([[cos(radians(strike-90)), -sin(radians(strike-90))], 
                [sin(radians(strike-90)), cos(radians(strike-90))]])

    # convert slip and rake to strike-slip and dip-slip
    ss=slip*cos(radians(rake))
    ds=slip*sin(radians(rake))

    # how many data points are we dealing with?
    n = len(data)
    
    UX = np.zeros(n)
    UY = np.zeros(n)
    UZ = np.zeros(n)

    for i in range(n):
    
        # shift and rotate the coordinates into Okada geometry
        P=np.array([[data[i,0]-xc],[data[i,1]-yc]]); # observation point wrt centroid in map coordinates
        Q=R.dot(P)                         # observation point rotated into Okada geometry
            
        # run the Okada dc3d function on the rotated coordinates   
        success, u, grad_u = dc3dwrapper(alpha,
                                            [Q[0], Q[1], 0],
                                            cd_depth, dip,
                                            [-as_length/2, as_length/2], 
                                            [-dd_width/2, dd_width/2],
                                            [ss, ds, 0.0])
        assert(success == 0)
        
        # here u[0] is strike-parallel displacement and u[1] is strike-normal displacement
        UX[i] = u[0]*sin(radians(strike))-u[1]*cos(radians(strike))   # x displacement
        UY[i] = u[0]*cos(radians(strike))-u[1]*sin(radians(strike))   # y displacement
        UZ[i] = u[2]   # z displacement
    
    ULOS = np.multiply(UX,data[:,3]) + np.multiply(UY,data[:,4]) + np.multiply(UZ,data[:,5])
    
    return ULOS

def los_penalty_fault(fparams, eparams, data):
    "Calculates total squared penalty for an Okada rectangular fault model, removing the best-fitting zero level shift"
    
    # same inputs as for rect_shear_fault - this time we'll calculate a penalty function from the model
    
    # calculate the model
    model_los_disps = rect_shear_fault(fparams, eparams, data)
    
    # estimate the mean residual
    zero_shift = np.mean(data[:,2]-model_los_disps)
    
    # subtract it when computing the total squared penalty
    penalty = np.sum(np.square((data[:,2]-zero_shift)-model_los_disps))
    
    return penalty
