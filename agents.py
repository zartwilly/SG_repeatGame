#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:08:11 2024

@author: willy

agents file contains functions and classes for items that are independant actions 
in the game
"""
import numpy as np
from enum import Enum

#%% state
class State(Enum): # Define possible states for a prosumer
    DEFICIT = "DEFICIT" 
    SELF = "SELF"
    SURPLUS = "SURPLUS"
    
#%% mode
class Mode(Enum):  # Define possible modes for a prosumer
    CONSPLUS = "CONS+"
    CONSMINUS = "CONS-"
    DIS = "DIS"
    PROD = "PROD"
    
#%% prosumer
class Prosumer:
    
    # This class represent a prosumer or an actor of the smartgrid

    state = None  # State of the prosumer for each period , possible values = {DEFICIT,SELF,SURPLUS}
    production = None # Electricity production during each period
    consumption = None # Electricity consumption during each period
    prodit = None # Electricity inserted to the SG
    consit = None # Electricity consumed from the SG
    storage = None # Electricity storage at the beginning of each period
    smax = None # Electricity storage capacity
    gamma = None # Incentive to store or preserve electricity
    phi = None # Random variable used to calculate gamma
    mode = None # Mode of the prosumer for each period, possible values = {CONSPLUS,CONSMINUS,DIS,PROD}
    prmode = None # Probability to choose between the two possible mode for each state at each period shaped like prmode[period][mode] (LRI only)
    virtualben = None # Benefit inside intern virtual economic system  for each period
    virtualcost = None # Cost inside intern virtual economic system for each period
    utility = None # Result of utility function for each period
    minbg = None # Minimum benefit obtained during all periods
    maxbg = None # Maximum benefit obtained during all periods
    benefit = None # Benefits for each period
    
    ##### new parameters variables for Repeated game ########
    Obj = None # Objective function
    price = None # price by by each prosumer during all periods
    valOne = None #
    Repart = None # a repartition function based on shapley value
    benefit = None # reward for each prosumer at each period
    

    def __init__(self, maxperiod, initialprob):
        """
        maxperiod : explicit ; 
        initialprob : initial value of prmode[0]
        """
        
        self.state = np.zeros(maxperiod, dtype=State)
        self.production = np.zeros(maxperiod) 
        self.consumption = np.zeros(maxperiod)
        self.prodit = np.zeros(maxperiod)
        self.consit = np.zeros(maxperiod)
        self.storage = np.zeros(maxperiod) 
        self.smax = 0
        self.gamma = np.zeros(maxperiod) 
        self.phi = np.zeros(maxperiod)
        self.mode = np.zeros(maxperiod, dtype=Mode)
        self.prmode = np.zeros((maxperiod,2))
        for i in range(maxperiod):
            self.prmode[i][0] = initialprob
            self.prmode[i][1] = 1 - initialprob
        self.virtualben = np.zeros(maxperiod)        
        self.virtualcost = np.zeros(maxperiod) 
        self.utility = np.zeros(maxperiod)
        self.benefit = np.zeros(maxperiod)
        
        ##### new parameters variables for Repeated game ########
        self.ObjValue = np.zeros(maxperiod)
        #self.ant = np.zeros(maxperiod)
        self.price = np.zeros(maxperiod)
        self.Repart = np.zeros(maxperiod)
        self.valOne = np.zeros(maxperiod)
        self.benefit = 0

