# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:11:35 2019

@author: Tish
"""

import numpy as np
from scipy.special import expit
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from sklearn.utils import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

class Naive_Bayes():
    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.classes = set(self.ytrain)
        self.mean = np.mean(self.xtrain)
        self.std = np.std(self.xtrain)
        self.predictions = []
        self.scores = []
    
    def train(self):
        self.class_mean= np.zeros((len(self.classes), self.xtrain.shape[1]))
        self.class_std = np.zeros((len(self.classes), self.xtrain.shape[1]))
        self.prior = np.zeros((len(self.classes),))
        
        for c in self.classes:
            indx = np.where(self.ytrain == c)
            self.prior[c] = indx[0].shape[0] / float(self.ytrain.shape[0])
            self.class_mean[c] = np.mean(self.xtrain[indx], axis=0)
            self.class_std[c] = np.std(self.xtrain[indx], axis=0)
        
    def predict(self, xtest):
        for xi in xtest:
            tiles = np.repeat([xi], len(self.classes), axis=0)
            E = norm.pdf((self.mean - xi) / self.std) 
            E = np.prod(E)
            
            self.likelihood = norm.pdf((tiles - self.class_mean) / self.class_std)
            self.likelihood = np.prod(self.likelihood, axis=1)
            
            self.post = self.prior * self.likelihood / E
            self.scores.append(1-max(self.post))
            self.predictions.append(np.argmax(self.post))
                
    def accuracy_score(self, ytrue):
        correct = 0
        for i in range(len(ytrue)):
            if ytrue[i] == self.predictions[i]:
                correct += 1
        return (correct/float(len(ytrue))) * 100.0
    
    def get_results(self, ylabel, kFold=False, test=True):
        self.ylabel = ylabel
        thresholds = []
        if kFold:
            conf = confusion_matrix(self.ylabel, self.predictions[18000:])
            plt.figure(0).clf()
            plt.imshow(conf)
            print(classification_report(self.ylabel, self.predictions[:18000]))
            fpr, tpr, thresholds = roc_curve(self.ylabel, self.scores[:18000])
            auc = roc_auc_score(self.ylabel, self.predictions[:18000])
        elif test:
            size = len(ylabel)
            conf = confusion_matrix(self.ylabel, self.predictions[-size:])
            plt.figure(0).clf()
            plt.imshow(conf)
            print(classification_report(self.ylabel, self.predictions[-size:]))
            fpr, tpr, thresholds = roc_curve(self.ylabel, self.scores[-size:])
            auc = roc_auc_score(self.ylabel, self.predictions[-size:])
        else:
            conf = confusion_matrix(self.ylabel, self.predictions)
            plt.figure(0).clf()
            plt.imshow(conf)
            print(classification_report(self.ylabel, self.predictions))
            fpr, tpr, thresholds = roc_curve(self.ylabel, self.scores)
            auc = roc_auc_score(self.ylabel, self.predictions)
        plt.figure(1).clf()

        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()

    
def main(csvX, csvY, kFold=False, boosting=False): 
    dfA = pd.read_csv(csvX)
    dfy = pd.read_csv(csvY)
    dfA = dfA[1:]
       
    names = dfA.columns 
    if 'domain1' in names:
        dfA = dfA.drop(columns=['domain1'])
        dfA = dfA.drop(columns=['state1'])
    else:
        dfA = dfA.drop(columns=['state1'])
        dfA = dfA.drop(columns=['custAttr2'])
 
    dfA = pd.get_dummies(dfA)
    
    X = np.array(dfA)
    y = np.array(dfy)
    pca = PCA(n_components=17)
    pca_vals = pca.fit_transform(X)
    V = pca.components_
    pca_X = np.matmul(X, V[:6, :].T)   
    scaler = StandardScaler()        
    pca_X = scaler.fit_transform(pca_X)

    mask = np.random.rand(len(X)) < 0.8
    
    idx = np.random.permutation(list(range(X.shape[0])))
    if pca:
        pca_X = pca_X[idx, :]
        y = y[idx]
        trX = pca_X[mask]
        testX = pca_X[~mask]
        trY = y[mask]
        testY = y[~mask] 
    else:
        X = X[idx, :]
        y = y[idx]
        trX = X[mask]
        testX = X[~mask]
        trY = y[mask]
        testY = y[~mask] 
        
    trY = np.squeeze(trY)
    testY = np.squeeze(testY)

    ###### 1.1 ######
    if kFold:
        ### train set ###
        size = 18000
        idx = np.random.permutation(list(range(X.shape[0])))
        if pca:
            pca_X = pca_X[idx, :]
            y = y[idx]
            pca_X = pca_X[:90000,:]
            y = y[:90000,:]
        else:
            X = X[idx, :]
            y = y[idx]
            X = X[:90000, :]
            y = y[:90000, :]
        for i in range(4):
            classifier = Naive_Bayes()
            l1 = pca_X[-size:,:]
            l2 = pca_X[:-size,:]
            pca_X = np.vstack((l1,l2)) 
            testX = pca_X[-size:,:]
            testY = y[-size:,:]
            trainX = pca_X[:-size,:]
            trainY = y[:-size,:]
            trainY = np.squeeze(trainY)
            #testY = np.squeeze(testY)

            for i in range(4):
                classifier.fit(trainX[i*size:(i+1)*size,:], trainY[i*size:(i+1)*size])
                classifier.train()
                classifier.predict(trainX[i*size:(i+1)*size,:])
                acc = classifier.accuracy_score(trainY[i*size:(i+1)*size])
                print("Train Accuracy: ", acc, "\tk: ", i+1)
            classifier.predict(testX)
            acc = classifier.accuracy_score(testY)
            print("Test Acc: ", acc)
            classifier.get_results(testY,kFold)
            plt.clf()
            plt.cla()
            plt.close()       
            
    else:
        classifier = Naive_Bayes()
        classifier.fit(trX, trY)
        classifier.train()
        classifier.predict(trX)
        acc = classifier.accuracy_score(trY)
        print("Train Accuracy: ", acc)
        
        
        classifier.predict(testX)
        acc = classifier.accuracy_score(testY)
        print("Test Accuracy: ", acc)
        classifier.get_results(testY,False)
        

            
if __name__ == '__main__':
    plt.ion()
    main(os.path.join(os.getcwd(), "Set-A.X.csv"),
         os.path.join(os.getcwd(), "Set-A.y.csv"))
    main(os.path.join(os.getcwd(), "Set-B.X.csv"),
         os.path.join(os.getcwd(), "Set-B.y.csv"))