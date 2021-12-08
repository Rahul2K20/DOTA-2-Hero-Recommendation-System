import numpy as np

def logistic_loss(a3, Y):
    m = Y.shape[0]
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    return cost
