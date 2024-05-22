#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:48:57 2024

@author: willy

application is the environment of the repeted game
"""


class App:
    
    """
    this class is used to call the various algorithms
    """
    
    SG = None # Smartgrid
    maxstep = None # Number of learning steps
    maxstep_init = None # Number of learning steps for initialisation Max, Min Prices. in this step, no strategies' policies are updated
    mu = None # To define
    b = None # learning rate / Slowdown factor LRI
    
    def __init__(self, maxstep, mu, b, maxstep_init):
        self.maxstep = maxstep
        self.mu = mu
        self.b = b
        self.maxstep_init = maxstep_init
        
        
    def run_LRI_4_onePeriodT_oneStepK(self, period, boolInitMinMax):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t.
        boolInitMinMax : Bool
            prescribe the game whether LRI probabilities strategies are updated or not.
            if True, LRI probabilities are not updated otherwise

        Returns
        -------
        None.

        """
        # Update prosumers' modes following LRI mode selection
        self.SG.updatemodeLRI(period, self.threshold)
        
        # Update prodit, consit and period + 1 storage values
        self.SG.updatesmartgrid(period)
        
        # Calculate inSG and outSG
        self.SG.computeSumInput(period)
        self.SG.computeSumOutput(period)
    
        # Calculate ValOne, ValEgo, ValLess, ValSG, Reduct
        self.SG.computeValOne(period)
        self.SG.computeValEgoc(period)
        self.SG.computeValLess(period)
        self.SG.computeValSG(period)
        self.SG.computeReduct(period)
        
        # Calculate Repart, Price, Obj, Ant for prosumers
        self.SG.computeRepart(period)
        self.SG.computePrice(period)
        self.SG.computeObjectiveValue(period)
        self.SG.computeAnt(period)
        
        # Update min/max prices for prosumers
        self.SG.updateMaxMinPrice(period)
        
        # boolInitMinMax == False, we update probabilities (prmod) of prosumers strategies
        if not boolInitMinMax:
            # Calculate utility
            self.SG.computeUtility(period)
            
            # Update probabilities for choosing modes
            self.SG.updateProbaLRI(period, self.b)
        
        pass
    
    def runLRI(self, file):
        """
        Run LRI algorithm with the repeated game
        
        Parameters
        ----------
        file : TextIO
            file to save some informations of runtime

        Returns
        -------
        None.

        """
        K = self.maxstep
        T = self.SG.maxperiod
        L = self.maxstep_init
        
        for t in range(T):
                        
            # Update the state of each prosumer
            self.SG.updateState(t)
            
            # Game for initialisation Min/Max Prices for prosumers
            for l in range(L):
                self.run_LRI_4_onePeriodT_oneStepK(t, boolInitMinMax=True)
                
            # Game with learning steps
            for k in range(K):
                self.run_LRI_4_onePeriodT_oneStepK(t, boolInitMinMax=False)
                
        # Compute Benifit Bi
        self.SG.computeBenefit()
        
        file.write("___Threshold___ \n")
        # Determines if the threshold has been reached
        N = self.SG.prosumers.size
        for t in range(T):
            for i in range(N):
                if (self.SG.prosumers[i].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[i].prmode[t][1]) < self.threshold):
                    file.write("Threshold not reached for period "+ str(i+1) +"\n") 
                    for Ni in range(N):
                        file.write("Prosumer " + str(Ni) + " : "+ str(self.SG.prosumers[Ni].prmode[i][0]) + "\n")
                    break
                
        # Determines for each period if it attained a Nash equilibrium and if not if one exist
        file.write("___Nash___ : NOT DEFINE \n")
                
            
            
                
                
                
                
                
        
        

