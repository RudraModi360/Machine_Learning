import numpy as np

def load_data():
    data=np.loadtxt("placement.txt",delimiter=',')
    X=data[:,0]
    y=data[:,1]
    return X,y
