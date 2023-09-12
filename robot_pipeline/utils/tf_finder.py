import numpy as np
import math

from scipy.optimize import minimize

def get_rot(xrot):
    c1=math.cos(xrot[0])
    s1=math.sin(xrot[0])
    c2=math.cos(xrot[1])
    s2=math.sin(xrot[1])
    c3=math.cos(xrot[2])
    s3=math.sin(xrot[2])
    rot=np.matrix([[ c2*c3,  c1*s3+s1*s2*c3, s1*s3-c1*s2*c3],\
                   [-c2*s3,  c1*c3-s1*s2*s3, s1*c3+c1*s2*s3],\
                   [ s2,    -s1*c2,          c1*c2 ]])
    return rot
    
def get_tf(x):
    rot=get_rot(x[:3])
    tf= np.matrix([[rot[0,0], rot[0,1], rot[0,2], x[3]],\
                   [rot[1,0], rot[1,1], rot[1,2], x[4]],\
                   [rot[2,0], rot[2,1], rot[2,2], x[5]],\
                   [0,        0,        0,        1]])
    
    return tf

def augment_data(data):
    v = np.ones((data.shape[0], 1))
    return np.c_[data, v]

def find_tf(data1,data2):
    def tf_error(x):
        """  sum(sum( (TF*D1-D2).^2))  """
        return  np.sum(np.sum(np.square((get_tf(x)*augment_data(data1).transpose()-augment_data(data2).transpose()))))
    x0 = np.array([0, 0, 0, 0, 0, 0])
    x1 = np.c_[(np.random.rand(10,3)*2-1)*math.pi,np.random.rand(10,3)*2-1]
    res = minimize(tf_error, x0, method='nelder-mead',options={'maxiter': 1000000, 'xtol': 1e-20, 'disp': False})
    current_fun=res.fun
    current_x=res.x
    for x in x1:
        res = minimize(tf_error, x, method='nelder-mead',options={'maxiter': 1000000, 'xtol': 1e-20, 'disp': False})
        if (res.fun<current_fun):
            current_fun=res.fun
            current_x=res.x

    print "Current function value: ", current_fun
    print get_tf(current_x)

    return get_tf(current_x)




