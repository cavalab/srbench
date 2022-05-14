from sklearn import ensemble

import sys, string, os, subprocess, time, logging
import glob 
import os.path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from subprocess import Popen, PIPE
from pathlib import Path
from sympy import *
from datetime import datetime
import re

# Settings
numberOfGen = 20# Default 100
sizeOfPopulation = 500 # Default 100
crossbreedingRate = 0.3 # Default 0.5
mutationRate = 0.3 # Default 0.5
depth = 5 # Default 5
filterOrigin = 1 # Default 1
tournamentSize = 3 #Default 5
numOfEquations = sizeOfPopulation
operationsToRemove = "max,min,absolute,arctan"
parallelOption = ""
fitDuration = 1
numberOfThreads = 6
exeName = "../submission/tree_genetic_programming/TreeGeneticProgramming"
logFolder = "logOutput"
loopWithSameDatafile = 3

# Time Efficiency Parameter
populationIncrement = 1000
timesToken = [#"Crossbreeding",
              "ChildrenCreation",
              "FitThread",
              "ParetoThread",
              "FilterThread",
              #"updateDistanceToData",
              #"updateEquations"
              ]

# Benchmark Output Option
isDisplayingFitnessChart = False
isCreatingCSV = False
isComputingTimeEfficiency = True
isThreadTesting = False



def launchCmd (inputPathSelected, inputDataFileSelected) :
    # Parameters
    ### Remove if you don't want numOfEquations = sizeOfPopulation ###
    numOfEquations = sizeOfPopulation
    ###
    paramSilent = " -s"
    paramNbGen = " -g " + str(numberOfGen)
    paramSizePop = " -p " + str(sizeOfPopulation)
    paramCbRate = " -c " + str(crossbreedingRate)
    paramMutRate = " -m " + str(mutationRate)
    paramDepth = " -e " + str(depth)
    paramFilterOrigin = " -f " + str(filterOrigin)
    paramNumOfEquations = " -n " + str(numOfEquations)
    paramsOperations = " -o " + operationsToRemove
    paramParallel = parallelOption
    paramsFitDuration = " -fd " + str(fitDuration)
    paramsNumOfThreads = " -nt " + str(numberOfThreads)


    parametersList = paramSilent + paramNbGen + paramSizePop + paramCbRate
    parametersList += paramMutRate + paramDepth + paramFilterOrigin + paramNumOfEquations
    parametersList += paramsOperations + paramParallel + paramsFitDuration + paramsNumOfThreads
                    
    paramInputData = " -i "

    # No debug logging outpout in the application
    os.environ["QT_LOGGING_RULE"] = "*.debug=false"
    os.environ["QT_LOGGING_RULE"] = "*.warning=false"
    os.environ["QT_LOGGING_RULE"] = "*.info=false"

    commandExe = exeName # + parametersList + paramInputData + inputPathSelected + inputDataFileSelected
 

    # Launch .exe + Stockage equation
    cmdOutput = subprocess.check_output(commandExe, shell=False, text=True)
    
  
    return cmdOutput


def extractResultFromOutput(data):
    # Extract best equation
    equationToken = "=> First equation : "
    errorToken = " Distance : "
    computeTimeToken = " Compute Time : "
    generationToken = " Last Generation : "

    equationPos = data.rfind(equationToken)
    if equationPos != -1:
        errorPos = data.rfind(errorToken)
        timePos = data.rfind(computeTimeToken)
        generationPos = data.rfind(generationToken)

        equation = data[(equationPos + len(equationToken)):errorPos]
        error = data[(errorPos + len(errorToken)):timePos]

        # Convert time value in second with 3 digits
        timeValue = float(data[(timePos + len(computeTimeToken)):generationPos]) / 1000
        timeValue = float("{:.3f}".format(timeValue))

        generation = data[(generationPos + len(generationToken)):]

        return (equation, error, timeValue, generation)

print(os.getcwd())
def ls(path):
    for files, dirs in os.walk():
        for f in files:
            print(f)
        for d in dirs:
            print(d) 
ls('../')

msglog= launchCmd(sys.argv[0],sys.argv[1])
est = extractResultFromOutput(msglog)

def complexity(est):
    return len(est.best_estimator_)

def model(est): 
    return est.stack_2_eqn(est.best_estimator_)
