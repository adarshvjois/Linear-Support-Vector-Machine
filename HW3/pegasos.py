'''
Created on 20-Feb-2015

@author: adarsh
'''
from counting_sparse_vec import counting_sparse_vec
from util import increment, dotProduct
from time import time
from math import sqrt
import numpy as np

def hinge_loss(Xi, w, yi):
    ydotwx = yi * w.dot(Xi)
    return max(0, 1 - ydotwx)

def svm_loss(Xi, w, yi, lambda_reg):
    return (lambda_reg / 2) * w.norm_squared + hinge_loss(Xi, w, yi)

def gradient_checker(Xi, yi, lambda_reg, computed_w,
                     objective_func, gradient_func,
                     epsilon=0.01, tol=1e-4):
    
    approx_w = counting_sparse_vec()
    
    for key in computed_w:
        computed_w[key] += epsilon
        L_plus = svm_loss(Xi, computed_w, yi, lambda_reg)
        computed_w[key] -= 2 * epsilon
        L_minus = svm_loss(Xi, computed_w, yi, lambda_reg)
        approx_w[key] = (L_plus - L_minus) / (2 * epsilon)
        computed_w[key] += epsilon
        
    return sqrt((approx_w - computed_w).norm_squared) < tol
        
def prediction_error(X_test, y_test, w):
    predictions = [dotProduct(x, w) for x in X_test]
    c = float(0)
    for p, y in zip(predictions, y_test):
        if p * y < 0:
            c += 1.0
    return c / len(y_test)
                
def pegasos(X_train, y_train, lambda_reg=10 ** -3, epochs=5):
    '''
    The Quicker variant of the pegasos, pretty much 
    exactly as here:
    http://research.microsoft.com/pubs/192769/tricks-2012.pdf
    '''
    
    w = counting_sparse_vec()  # initial weight vector
    M = len(X_train)  # number of samples
    st = 1.0
    t = 1.0  # iteration numer
    
    for elem in X_train[0]:
        w[elem] = 1.0  # Just preallocating some values so that it speeds up first few iterations
    
    print "Lambda = ", lambda_reg
    gamma0 = 0.1
    for ep in range(epochs):
        for i in range(M):
            t = t + 1.0  # iteration number
            
            #gammat = 1.0 / (lambda_reg * t)  # step size
            gammat = gamma0 / (1 + gamma0 * lambda_reg * t)
            ywdotx = st * y_train[i] * w.dot(X_train[i])  # projected value scaled        
            
            st = (1.0 - gammat * lambda_reg * 1.0) * st  # scale            
            
            if ywdotx < 1:
                s = (gammat * y_train[i]) / (st * 1.0)
                w.scale_and_increment(s, X_train[i])
            
        # print "Epoch number::", ep
    return st * w

def pegasos_slow(X, y, lambda_reg=10 ** -3, epochs=5):
    t = 1
    M = len(X)
    w = counting_sparse_vec()
    
    for elem in X[0]:
        w[elem] = 1.0    
    for ep in range(epochs):
        
        for i in range(M):
            
            eta = 1 / (lambda_reg * t)
            
            t = t + 1.0
            
            ydotwx = y[i] * dotProduct(w, X[i])
            w = scale_vec((1.0 - eta * lambda_reg), w)
            
            if ydotwx < 1:
                increment(w, (eta * float(y[i])), X[i])
        
        print "epoch no:" + str(ep)
        print len(w)
    return w

def scale_vec(scale, vect):
    for key in vect:
        vect[key] *= scale
    return vect

def test_lambdas(X_train, y_train, X_test, y_test):
    prediction_errs = []
    t0 = time()
    lambdas = 2.0 ** np.arange(-15, 3,1)
    for l in lambdas:
        w = pegasos(X_train, y_train, lambda_reg=l, epochs=500)
        p = prediction_error(X_test, y_test, w)
        print p
        prediction_errs.append(p)
    t1 = time()
    print "Time taken:: " + str(t1 - t0)
    return prediction_errs
