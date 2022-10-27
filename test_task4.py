"""
This file contains tests that you can run to check your code.
In a terminal, navigate to this folder and run 

    pytest

to run the tests. For a more detailed output run

    pytest -v

or, to stop at the first failed test:

    pytest -x

More information can be found here: https://docs.pytest.org/en/7.1.x/reference/reference.html#command-line-flags

You are not supposed to understand or edit this file.

EDITING THIS FILE WILL NOT FIX THE PROBLEMS IN YOUR CODE!

"""


import numpy as np
import pytest

# use matplotlib backend that does not show any figures
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*Matplotlib.*")  # ignore warnings from matplotlib (due to backend)
warnings.filterwarnings("ignore", message=".*invalid value*")  # ignore warnings from numpy (due to div by zero)
warnings.filterwarnings("ignore", message=".*Mean of empty slice*")  # ignore warnings from numpy (due to empty array)

# import (and run) student script
import task4 as studentScript

# import reference results
refResults: dict = np.load("exercise1_refSol.npz")

## helper functions ##
# warning logger
def logWarning(message):
    warnings.warn(message)

# check whether two arrays are the same
def checkSimilar(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(np.isclose(a,b,equal_nan=True))

#############################
### GENERAL TESTS ###########
#############################

@pytest.fixture(scope="session", autouse=True)
def test_general():
    # a general test to guide the students, does not give points, and should only print warningss

    # check if all names exist
    for key in refResults.keys():
        # logWarning(key)
        if not hasattr(studentScript, key):  logWarning(f"The variable {key} does not exist.")

    # check all figures for legends and labels
    # for k in plt.get_fignums():
    #     fig = plt.figure(k)
    #     if fig.gca().get_xlabel() != "":  logWarning(f"There is no x-label in figure {k}.")
    #     if fig.gca().get_ylabel() != "": logWarning(f"There is no y-label in figure {k}.") 
    #     if fig.gca().get_legend() is not None:  logWarning(f"There is no legend on figure {k}.")

#############################
### TESTS FOR PART A ########
#############################

# check if computed correctly (3 points)

print('--------------------=====================')
def test_A_1():
    assert checkSimilar(refResults['R_SA_single'],studentScript.R_SA_single)
def test_A_2():
    assert checkSimilar(refResults['R_LS_single'],studentScript.R_LS_single)
def test_A_3():
    assert checkSimilar(refResults['R_EV_single'],studentScript.R_EV_single)

# check if plotted correctly (1 point)
def test_A_4():
    assert plt.fignum_exists(1), "Figure 1 does not not exist, make sure it is created with plt.figure(1)."
    fig = plt.figure(1)
    numLines = len(fig.gca().lines)
    assert numLines == 3, f"Figure 1 should show three lines, instead there are {numLines}."


#############################
### TESTS FOR PART B ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_B_1():
    
    # check if values correct
    assert checkSimilar(refResults['R_SA'],studentScript.R_SA)
    assert checkSimilar(refResults['R_EV'],studentScript.R_EV)
    assert checkSimilar(refResults['R_LS'],studentScript.R_LS)

    # check if three figures
    for k in [2,3,4]:

        # check if created
        assert plt.fignum_exists(k), f"Figure {k} does not not exist, make sure it is created with plt.figure({k})."

        # get figures
        fig = plt.figure(k)

        # check if figure show the right number of lines
        numLines = len(fig.gca().lines)
        targetNumLines = refResults['M']
        assert  numLines == targetNumLines, f"Figure {k} should show {targetNumLines} lines, instead there are {numLines}."

#############################
### TESTS FOR PART C ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_C_1():
    
    # check if values correct
    assert checkSimilar(refResults['R_SA_mean'],studentScript.R_SA_mean)
    assert checkSimilar(refResults['R_EV_mean'],studentScript.R_EV_mean)
    assert checkSimilar(refResults['R_LS_mean'],studentScript.R_LS_mean)


#############################
### TESTS FOR PART D ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_D_1():
    
    # check if values correct
    assert checkSimilar(refResults['R_SA_Nmax'],studentScript.R_SA_Nmax)
    assert checkSimilar(refResults['R_EV_Nmax'],studentScript.R_EV_Nmax)
    assert checkSimilar(refResults['R_LS_Nmax'],studentScript.R_LS_Nmax)
