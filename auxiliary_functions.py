#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:03:54 2024

@author: willy

auxiliary_functions file contains all functions that are used to all project files.
"""

def apv(x): # Return absolute positive value 
    return max(0,x)  
    
def phiepoplus(x): # Parameter :  benefit function of selling energy to EPO
    return x * 15

def phiepominus(x): # Parameter :  cost function of buying energy from EPO 
    return x * 90

