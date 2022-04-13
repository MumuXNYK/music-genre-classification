""" MACHINE LEARNING WITH PYTHON 
    MUSIC GENRE CLASSIFICATION USING K-NEAREST NEIGHBOR ALGORITHM 
    Author @MUMUKSH NAYAK """


# import the required modules and packages
from collections import defaultdict
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile

import os
import pickle
import random
import operator
import math


# Function To Perform Actual Distance Calculation Between Features
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(),
                 np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


# Function To Get The Distance Between Feature Vectors And Find Neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + \
            distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


# Identify The Class Of The Instance
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(),
                    key=operator.itemgetter(1), reverse=True)

    return sorter[0][0]


# Function To Evaluate The Model
def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1

    return (1.0 * correct) / len(testSet)


# Directory Of The Dataset
directory = "D:\python\Music_Genre_Classification\Data\genres/"

# Binary File To Collect Extracted Features
f = open("my.dat", 'wb')

i = 0

for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):
        try:
            (rate, sig) = wav.read(directory+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
        except Exception as e:
            print('Got an exception: ', e, ' in folder: ',
                  folder, ' filename: ', file)

f.close()


# Splitting Test And Training Dataset
dataset = []


def loadDataset(filename, split, trSet, teSet):
    with open('my.dat', 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


trainingSet = []
testSet = []
loadDataset('my.dat', 0.66, trainingSet, testSet)


# Making Predictions Using KNN
leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)


# Result Dictionary To Map The Genre Name With Genre ID
results = defaultdict(int)

directory = "D:\python\Music_Genre_Classification\Data\genres/"

i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1


# Testing The Code With Audio Samples
test_dir = "D:\python\Music_Genre_Classification\Test/"
test_file = test_dir + "test2.wav"


# Extracting The Features From The Test File
(rate, sig) = wav.read(test_file)
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, i)


# Predict The Result
pred = nearestClass(getNeighbors(dataset, feature, 5))
print(results[pred])
