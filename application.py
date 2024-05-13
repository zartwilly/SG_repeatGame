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
    mu = None # To define
    
    def __init__(self, maxstep, mu):
        self.maxstep = maxstep
        self.mu = mu
        
        
    def runLRI(self):
        """
        Run LRI algorithm with the repeated game

        Returns
        -------
        None.

        """
        L = self.maxstep
        T = self.SG.maxperiod
        
        

