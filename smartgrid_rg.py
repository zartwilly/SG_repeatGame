#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:29:28 2024

@author: willy

smartgrid_rg is the smartgrid in the repeat game that centralizes the parameters of the environment
"""
import math
import random as rdm
import numpy as np
import agents as ag
import auxiliary_functions as aux



class Smartgrid :
    
    # This class represent a smartgrid
    
    maxperiod = None # Number of periods
    prosumers = None # All prosumers inside the smartgrid
    bgmax = None # Maximum benefit for each prosumer for each period
    bgmin = None # Minimum benefit for each prosumer for each period
    piepoplus = None # Unitary price of electricity purchased by EPO
    piepominus = None # Unitary price of electricity sold by EPO
    insg = None # Sum of electricity input from all prosumers
    outsg = None # Sum of electricity outputs from all prosumers
    piplus = None # Unitary benefit of electricity sold to SG (independently from EPO)
    piminus = None # Unitary cost of electricity bought from SG (independently from EPO)
    betaplus = None # Intermediate values for computing piplus and real benefit 
    betaminus = None # Intermediate values for computing piminus and real cost
    unitaryben = None # Unitary benefit of electricity sold to SG (possibly partially to EPO)
    unitarycost = None # Unitary cost of electricity bought from SG (possibly partially from EPO)
    czerom = None # Upper bound a prosumer could have to pay
    realprod = None # Real value of production for each prosumers (different from the predicted production)
    realstate = None # Real state of each prosumers when using real production value (can be the same as the one determined with predicted production)
    
    ##### new parameters variables for Repeated game ########
    reduct = None #
    valsg = None #
    valEgoc = None #
    ant = None #  measures th deviation of the strategies (prmode) taken by each prosumer (in the SG) from the best one, in a context of EPO
    prices_min = None # Minimum price for each prosumer for each period
    prices_max = None # Maximum price for each prosumer for each period
    
    
    
    def __init__(self, N, maxperiod, initialprob, mu): 
        """
        N = number of prosumers, 
        initialprob : initial value of probabilities for LRI
        mu: parameter [0,1]
        
        """
        self.prosumers = np.ndarray(shape=(N), dtype=ag.Prosumer)
        self.maxperiod = maxperiod
        self.mu = mu
        for i in range(N):
            self.prosumers[i] = ag.Prosumer(maxperiod, initialprob)   
        self.bgmax = np.zeros((N, maxperiod))
        self.bgmin = np.zeros((N, maxperiod))
        self.piepoplus = np.zeros(maxperiod)    
        self.piepominus = np.zeros(maxperiod)     
        self.insg = np.zeros(maxperiod)       
        self.outsg = np.zeros(maxperiod)
        self.piplus = np.zeros(maxperiod)
        self.piminus = np.zeros(maxperiod)
        self.betaplus = np.zeros(maxperiod)
        self.betaminus = np.zeros(maxperiod)
        self.unitaryben = np.zeros(maxperiod)      
        self.unitarycost = np.zeros(maxperiod)
        self.czerom = np.zeros(maxperiod)
        self.realprod = np.zeros((N,maxperiod))
        self.realstate = np.zeros((N,maxperiod),dtype=ag.State)
        
        ##### new parameters variables for Repeated game ########
        self.reduct = np.zeros(maxperiod)
        self.valsg = np.zeros(maxperiod)
        self.ant = np.zeros(maxperiod)
        self.valEgoc = np.zeros(maxperiod)
        self.valLess = np.zeros(maxperiod)
        self.prices_min = np.zeros((N, maxperiod))
        self.prices_max = np.zeros((N, maxperiod))
        
      
    #%% new function for SG with repeated game
    ###########################################################################
    #                       compute smartgrid variables
    ###########################################################################
    def computeValSG(self, period):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        out_sg = self.outsg[period]
        int_sg = self.insg[period]
        
        self.valsg[period] = aux.phiepominus(out_sg - int_sg) \
                        - aux.phiepoplus(int_sg - out_sg)
        
    def computeValLess(self, period):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
            
        self.valLess[period] = aux.phiepominus(self.outsg[period]) \
                                - aux.phiepoplus(self.insg[period])
        
    
    def computeValOne(self, period):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        for i in range(self.prosumers.size):
            valOnei = None
            Cit = self.prosumers[i].consumption[period]
            Pit = self.prosumers[i].production[period]
            Sit = self.prosumers[i].storage[period]
            Smax = self.prosumers[i].smax
            valOnei = aux.phiepominus(aux.apv(Cit - Pit - Sit)) \
                        - aux.phiepoplus(aux.apv(Pit - Cit- (Smax - Sit)))
                        
            self.prosumers[i].valOne[period] = valOnei
        
        
    def computeValEgoc(self, period):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        sumValEgoc = 0
        for i in range(self.prosumers.size):
            sumValEgoc += self.prosumers[i].valOne[period]
            
        self.valEgoc[period] = sumValEgoc
    
    def computeReduct(self, period):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        self.reduct[period] = self.valEgoc[period] \
                                - self.valsg[period] \
                                - aux.phiepoplus(self.insg[period])
        
    
    def computeRepart(self, period):
        """
        compute a repartition value of each prosumer

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        N = self.prosumers.size
        
        for i in range(N):
            self.prosumers[i].Repart[period] \
                = self.mu \
                    * (self.reduct[period]/N) \
                    + (1-self.mu) \
                        * (self.reduct[period] * self.prosumers[i].prodit[period] \
                           / self.insg[period])
        
        pass
    
    def computePrice(self, period):
        """
        compute the price applying to each prosumer

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        
        for i in range(self.prosumers.size):
            self.prosumers[i].price[period] \
                = aux.phiepominus(self.prosumers[i].consit[period]) \
                    - aux.phiepoplus(self.prosumers[i].prodit[period]) \
                    + aux.prosumers[i].Repart[period]
                    
    def computeObjectiveValue(self, period):
        """
        compute the objective value for each prosumers

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].ObjValue[period] \
                = self.prosumers[i].valOne[i] - self.prosumers[i].price[i]
                
                
    def computeAnt(self, period):
        """
        measures th deviation of the strategies (prmode) taken by each 
        prosumer (in the SG) from the best one, in a context of EPO


        Parameters
        ----------
        period : int
             an instance of time t

        Returns
        -------
        None.

        """
        self.ant[period] = self.valLess[period] - self.valEgoc[period]
       
    def computeSumInput(self, period): 
        """
        Calculate the sum of the production of all prosumers during a period
        
        Parameters
        ----------
        period: int 
            an instance of time t
        """
        tmpsum = 0
        for i in range(self.prosumers.size):
            tmpsum = tmpsum + self.prosumers[i].prodit[period]
        self.insg[period] = tmpsum
    
    def computeSumOutput(self, period): 
        """
        Calculate sum of the consumption of all prosumers during a period
        
        Parameters
        ----------
        period: int 
            an instance of time t
        """
        tmpsum = 0
        for i in range(self.prosumers.size):
            tmpsum = tmpsum + self.prosumers[i].consit[period]
        self.outsg[period] = tmpsum

    def computeUtility(self, period): 
        """
        Calculate utility function using min, max and last prosumer's benefits
        
        Parameters
        ----------
        period: int 
            an instance of time t
            
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prices_max[i][period] != 0 or self.prices_min[i][period] != 0 :
                nume = self.prices_max[i][period] - self.prosumers[i].price[period]
                demo = self.prices_max[i][period] - self.prices_min[i][period]
                self.prosumers[i].utility[period] = 1 - nume/demo

            else:
                self.prosumers[i].utility[period] = 0
                
    def computeBenefit(self): 
        """
        Calculate benefit for each prosumer for all period
        
        Parameters
        ----------
        period: int 
            an instance of time t
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            self.prosumers[i].benefit = np.sum(self.prosumers[i].price)
                
    ###########################################################################
    #                       update prosumers variables
    ###########################################################################
    def updateMaxMinPrice(self, period):
        """
        update the max and the min prices for each prosumer at one period

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        for i in range(self.prosumers.size):
            self.prices_max[i][period] = max(self.prices_max[i][period], 
                                             self.prosumers[i].price[period])
            self.prices_min[i][period] = min(self.prices_min[i][period], 
                                             self.prosumers[i].price[period])
    
    def updateState(self, period): 
        """
        Change prosumer's state based on its production, comsumption and available storage
        
        Parameters
        ----------
        period : int 
            an instance of time t
        """
        N = self.prosumers.size
        
        for i in range(N):    
            if self.prosumers[i].production[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] = ag.State.SURPLUS
            
            elif self.prosumers[i].production[period] \
                + self.prosumers[i].storage[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] = ag.State.SELF
            
            else :
                self.prosumers[i].state[period] = ag.State.DEFICIT
                
    def updatemodeLRI(self, period, threshold): 
        """
        # Update mode using rules from LRI
        
        Parameters:
        ----------
        period : int
            an instance of time t
        
        threshold: float
            an threshold 
        """
        N = self.prosumers.size
        
        for i in range(N):
            rand = rdm.uniform(0,1)
            
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                if (rand <= self.prosumers[i].prmode[period][0] \
                    and self.prosumers[i].prmode[period][1] < threshold) \
                    or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.PROD
            
            elif self.prosumers[i].state[period] == ag.State.SELF :
                if (rand <= self.prosumers[i].prmode[period][0] \
                    and self.prosumers[i].prmode[period][1] < threshold) \
                    or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
            
            else :
                if (rand <= self.prosumers[i].prmode[period][0] \
                    and self.prosumers[i].prmode[period][1] < threshold) \
                    or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                    
    def updatesmartgrid(self, period): 
        """
        Update storage for next period ie period+1, consit, prodit based on mode and state
        
        Parameters:
        ----------
        period : int
            an instance of time t
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT:
                self.prosumers[i].prodit[period] = 0
                if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS:
                    self.prosumers[i].storage[period+1] = 0
                    self.prosumers[i].consit[period] \
                        = self.prosumers[i].consumption[period] \
                            - (self.prosumers[i].production[period] \
                               + self.prosumers[i].storage[period])
                
                else :
                    self.prosumers[i].storage[period+1] = self.prosumers[i].storage[period]
                    self.prosumers[i].consit[period] \
                        = self.prosumers[i].consumption[period] \
                            - self.prosumers[i].production[period]
            
            elif self.prosumers[i].state[period] == ag.State.SELF:
                self.prosumers[i].prodit[period] = 0
                
                if self.prosumers[i].mode[period] == ag.Mode.CONSMINUS:
                    self.prosumers[i].storage[period+1] = self.prosumers[i].storage[period]
                    self.prosumers[i].consit[period] \
                        = self.prosumers[i].consumption[period] \
                            - self.prosumers[i].production[period]
                
                else :
                    self.prosumers[i].storage[period+1] \
                        = self.prosumers[i].storage[period] \
                            - (self.prosumers[i].consumption[period] \
                               - self.prosumers[i].production[period])
                    self.prosumers[i].consit[period] = 0
            else :
                self.prosumers[i].consit[period] = 0
                
                if self.prosumers[i].mode[period] == ag.Mode.DIS:
                    self.prosumers[i].storage[period+1] \
                        = min(self.prosumers[i].smax,self.prosumers[i].storage[period] \
                              +(self.prosumers[i].production[period] \
                                - self.prosumers[i].consumption[period]))
                    self.prosumers[i].prodit[period] \
                        = aux.apv(self.prosumers[i].production[period] \
                              - self.prosumers[i].consumption[period] \
                                  -(self.prosumers[i].smax \
                                    - self.prosumers[i].storage[period] ))
                else:
                    self.prosumers[i].storage[period+1] = self.prosumers[i].storage[period]
                    self.prosumers[i].prodit[period] \
                        = self.prosumers[i].production[period] \
                            - self.prosumers[i].consumption[period]
   
    def updateProbaLRI(self, period, slowdown): 
        """
        Update probability for LRI based mode choice
        
        Parameters
        ----------
        period: int 
            an instance of time t
            
        slowdown: float
            Slowdown factor or learning rate
            
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                if self.prosumers[i].mode[period] == ag.Mode.DIS :
                    self.prosumers[i].prmode[period][0] \
                        = min(1,
                              self.prosumers[i].prmode[period][0] \
                                  + slowdown \
                                      * self.prosumers[i].utility[period] \
                                      * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] \
                        = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] \
                        = min(1,
                              self.prosumers[i].prmode[period][1] \
                                  + slowdown \
                                      * self.prosumers[i].utility[period] \
                                      * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] \
                        = 1 - self.prosumers[i].prmode[period][1]
                    
            elif self.prosumers[i].state[period] == ag.State.SELF:
                if self.prosumers[i].mode[period] == ag.Mode.DIS :
                    self.prosumers[i].prmode[period][0] \
                        = min(1,
                              self.prosumers[i].prmode[period][0] \
                                  + slowdown \
                                      * self.prosumers[i].utility[period] \
                                      * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] \
                        = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] \
                        = min(1,
                              self.prosumers[i].prmode[period][1] \
                                  + slowdown \
                                      * self.prosumers[i].utility[period] \
                                      * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] \
                        = 1 - self.prosumers[i].prmode[period][1]
            else :
                if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS :
                    self.prosumers[i].prmode[period][0] \
                        = min(1,
                              self.prosumers[i].prmode[period][0] \
                                  + slowdown \
                                      * self.prosumers[i].utility[period] \
                                      * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] \
                        = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] \
                        = min(1,
                              self.prosumers[i].prmode[period][1] \
                                  + slowdown \
                                      * self.prosumers[i].utility[period] \
                                      * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] \
                        = 1 - self.prosumers[i].prmode[period][1]
    
    
    #%% ALL of the following methods apply to all the prosumers of the smartgrid over one given period
    
    
    # def computesuminput(self, period): 
    #     """
    #     Calculate the sum of the production of all prosumers during a period
        
    #     Parameters
    #     ----------
    #     period: int 
    #         an instance of time t
    #     """
    #     tmpsum = 0
    #     for i in range(self.prosumers.size):
    #         tmpsum = tmpsum + self.prosumers[i].prodit[period]
    #     self.insg[period] = tmpsum
    
    # def computesumoutput(self, period): 
    #     """
    #     Calculate sum of the consumption of all prosumers during a period
        
    #     Parameters
    #     ----------
    #     period: int 
    #         an instance of time t
    #     """
    #     tmpsum = 0
    #     for i in range(self.prosumers.size):
    #         tmpsum = tmpsum + self.prosumers[i].consit[period]
    #     self.outsg[period] = tmpsum

    def computevirtualbenefit(self, period): 
        """
        Calculate total benefit inside the virtual economic system
        
        Parameters
        ----------
        period: int 
            an instance of time t
        
        """
        N = self.prosumers.size
        r = 0
        
        for i in range (N):
            
            if self.prosumers[i].mode[period] == ag.Mode.CONSMINUS :
                r = self.prosumers[i].storage[period]
                
            elif self.prosumers[i].mode[period] == ag.Mode.DIS \
                and self.prosumers[i].state[period] == ag.State.SURPLUS:
                r = min(self.prosumers[i].smax - self.prosumers[i].storage[period],\
                        self.prosumers[i].production[period] \
                            - self.prosumers[i].consumption[period])
                
            self.prosumers[i].virtualben[period] \
                = (self.unitaryben[period] * self.prosumers[i].prodit[period]) \
                    + (self.prosumers[i].gamma[period] * r)
    
    def computevirtualcost(self, period): 
        """
        Calculate total cost inside the virtual economic system
        
        Parameters
        ----------
        period: int 
            an instance of time t
        
        """
        
        N = self.prosumers.size
        for i in range(N):
            self.prosumers[i].virtualcost[period] \
                = self.unitarycost[period] * self.prosumers[i].consit[period]
            
    def computephi(self, period): 
        """
        Compute value of the random variable phi, used to compute gamma
        
        Parameters
        ----------
        period: int 
            an instance of time t
            
        """
        N = self.prosumers.size
        
        # Set splus and sminus depending of state
        for i in range(N):
            
            if self.prosumers[i].state[period] == ag.State.SURPLUS :
                splus = max(self.prosumers[i].smax,self.prosumers[i].storage[period] \
                            + (self.prosumers[i].production[period] \
                               - self.prosumers[i].consumption[period]))
                sminus = self.prosumers[i].storage[period]
                
            elif self.prosumers[i].state[period] == ag.State.SELF:
                splus = self.prosumers[i].storage[period]
                sminus = self.prosumers[i].storage[period] \
                            - (self.prosumers[i].consumption[period] \
                               - self.prosumers[i].production[period])   
            
            else: 
                splus = self.prosumers[i].storage[period]
                sminus = 0
        
            # Calculate pp probability
            if (splus - sminus) != 0 :
                
                if aux.apv(aux.apv(self.prosumers[i].consumption[period+1] \
                           - self.prosumers[i].production[period+1]) \
                       - sminus) / (splus - sminus) <= 0:
                    pp = 0
                
                else:
                    pp = math.sqrt(
                            min(1, aux.apv(aux.apv(self.prosumers[i].consumption[period+1] \
                                                   - self.prosumers[i].production[period+1]) \
                                           - sminus) / (splus - sminus)))
            else :
                
                if aux.apv(aux.apv(self.prosumers[i].consumption[period+1] \
                                   - self.prosumers[i].production[period+1]) \
                           - sminus) <= 0:
                    pp = 0
                
                else:
                    pp = math.sqrt(
                            min(1, 
                                aux.apv(aux.apv(self.prosumers[i].consumption[period+1] \
                                               - self.prosumers[i].production[period+1]) \
                                        - sminus)))
            
            # Calculate random variable phi
            rand = rdm.uniform(0,1)
            if  rand <= pp :
                self.prosumers[i].phi[period] = 1
                
            else :
                self.prosumers[i].phi[period] = 0
                

    def computegamma(self, period): 
        """
        Calculate the incentive to store electricity for a period
        
        Parameters
        ----------
        period: int 
            an instance of time t
            
        """
        N = self.prosumers.size
        # Fix X,splus,sminus depending on state value
        for i in range(N):
            
            if self.prosumers[i].state[period] == ag.State.SURPLUS :
                X = self.piplus[period]
            
            else:
                X = self.piminus[period]

            self.prosumers[i].gamma[period] = self.prosumers[i].phi[period] * (X + 1)

    def computepiepo(self, period) : 
        """
        Calculate unitary selling/buying price of an unit of electricity for 
        a period inside the real economic system
        
        Parameters
        ----------
        period: int 
            an instance of time t
            
        """
        qtplus = 0
        qtminus = 0
        N = self.prosumers.size
        
        for i in range(N):
            qtplus +=  aux.apv(self.prosumers[i].production[period] \
                               - self.prosumers[i].consumption[period]) \
                        - \
                        aux.apv(self.prosumers[i].consumption[period] \
                                - (self.prosumers[i].production[period] \
                                   + self.prosumers[i].storage[period]))
                
            qtminus += aux.apv(self.prosumers[i].consumption[period] \
                               - self.prosumers[i].production[period]) \
                        - \
                        aux.apv(self.prosumers[i].production[period] \
                                - (self.prosumers[i].consumption[period] \
                                   + self.prosumers[i].storage[period] \
                                   - self.prosumers[i].smax))
        
        qtplus = aux.apv(qtplus)
        qtminus = aux.apv(qtminus)
        
        if qtplus == 0 :
            self.piepoplus[period] = aux.phiepoplus(1)
        
        else :
            self.piepoplus[period] = aux.phiepoplus(qtplus)/qtplus
        
        if qtminus == 0 :
            self.piepominus[period] = aux.phiepominus(1)
        
        else:
            self.piepominus[period] = aux.phiepominus(qtminus)/qtminus
    
    def computebetaplus(self, period):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        soldvol = 0 # Total volume of electricity sold to EPO by SG at period period
        producedvol = 0 # Total produced volume of electricity of SG at period period
        
        for t in range(period):
            soldvol = soldvol + aux.apv(self.insg[t] - self.outsg[t])
            producedvol = producedvol + self.insg[t]
            
        soldvol = aux.apv(soldvol)
        
        if producedvol != 0:
            self.betaplus[period] = aux.phiepoplus(soldvol) / producedvol
        
        else :
            self.betaplus[period] = aux.phiepoplus(1)
            
    def computebetaminus(self, period):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        purchasedvol = 0 # Total volume of electricity purchased from EPO by SG at period period
        consummedvol = 0 # Total consummed volume of electricity of SG at period period
        
        for i in range(period):
            purchasedvol = purchasedvol + aux.apv(self.outsg[i] - self.insg[i])
            consummedvol = consummedvol + self.outsg[i]
        
        purchasedvol = aux.apv(purchasedvol)
        
        if consummedvol != 0:
            self.betaminus[period] = aux.phiepominus(purchasedvol) / consummedvol
        
        else :
            self.betaminus[period] = aux.phiepominus(1)
        
    # The next methods are both version of the same method using differents rules for updating piplus
    # The one used for LRI is computepiplusabmv which give the best results, consider testing both when using new algorithms
    
    def computepiplus(self, period):
        """
        Calculate selling price of a unit of electricity for a period inside 
        the real economic system
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.

        """
        if period < 2: # Cases where betaplus is not defined by last period
            if self.piepominus[period] != 0 :
                self.piplus[period] \
                    = ((self.piepoplus[1] - 1) \
                       / self.piepominus[period] ) * self.piepoplus[period]
            
            else : 
                self.piplus[period] \
                    = (self.piepoplus[1] - 1) * self.piepoplus[period]
        
        else:
            if self.piepominus[period] != 0 :
                self.piplus[period] \
                    = (self.betaplus[period - 1] \
                       / self.piepominus[period]) * self.piepoplus[period]
            
            else :
                self.piplus[period] \
                    = self.betaplus[period - 1] * self.piepoplus[period]
        
    def computepiplusabmv(self, period): 
        """
        Calculate selling price of a unit of electricity for a period inside 
        the real economic system with a different formula
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.
        """
        self.piplus[period] = self.piepoplus[period]
        
    # The next methods are both version of the same method using differents rules for updating piminus
    # The one used for LRI is computepiminusabmv which give the best results, consider testing both when using new algorithms
    
    def computepiminus(self, period): 
        """
        Calculate buying price of a unit of electricity for a period inside the real economic system
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.
        """
        if period < 2:
            self.piminus[period] = self.piepominus[1] - 1
        
        else:
            self.piminus[period] = self.betaminus[period -1]
        
    def computepiminusabmv(self, period): 
        """
        Calculate buying price of a unit of electricity for a period inside the real economic system with a different formula
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.
        """
        self.piminus[period] = self.piepoplus[period]
        
    def computeunitaryben(self, period): 
        """
        Calculate selling price of a unit of electricity for a period inside 
        the virtual economic system
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.
        """
        if self.outsg[period] > self.insg[period] :
            self.unitaryben[period] = self.piplus[period]
        else :
            self.unitaryben[period] \
                = (aux.phiepoplus(self.insg[period]  - self.outsg[period]) \
                   + self.outsg[period] * self.piplus[period]) \
                   / self.insg[period] 
    
    def computeunitarycost(self, period): 
        """
        Calculate buying price of a unit of electricity for a period inside 
        the virtual economic system
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.
        """
        if self.insg[period] >= self.outsg[period] :
            self.unitarycost[period] = self.piminus[period]
        
        else :
            self.unitarycost[period] \
                = (aux.phiepominus(self.outsg[period] - self.insg[period]) \
                   + self.insg[period] * self.piminus[period]) \
                   / self.outsg[period]
        
    def computeczerom(self, period): 
        """
        Calculate upper bound of prosumer cost to pay for a period
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        None.
        """
        im = 0
        iM = 0
        om = 0
        oM = 0
        N = self.prosumers.size
        
        # Computing the bounds used to compute czerom (im,IM,om,oM)
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                im = im + aux.apv(self.prosumers[i].consumption[period] \
                        + (self.prosumers[i].smax \
                           - self.prosumers[i].storage[period]))
                iM = iM + self.prosumers[i].production[period] \
                        - self.prosumers[i].consumption[period]
            
            if self.prosumers[i].state[period] == ag.State.DEFICIT:
                om = om + self.prosumers[i].consumption[period] \
                        - (self.prosumers[i].production[period] \
                           + self.prosumers[i].storage[period])
            
            if self.prosumers[i].state[period] == ag.State.DEFICIT \
                or self.prosumers[i].state[period] == ag.State.SELF:
                oM = oM + self.prosumers[i].consumption[period] \
                        - self.prosumers[i].production[period]
                
        if om != 0:
            self.czerom[period] = min(self.piminus[period], 
                                      (aux.phiepominus(oM - im) \
                                       + iM * self.piminus[period]) / om )
        
        else:
            self.czerom[period] = min(self.piminus[period], 
                                      (aux.phiepominus(oM - im) \
                                       + iM * self.piminus[period]))
            
    # def computeutility(self, period): 
    #     """
    #     Calculate utility function using min, max and last prosumer's benefits
        
    #     Parameters
    #     ----------
    #     period: int 
    #         an instance of time t
            
    #     """
    #     N = self.prosumers.size
        
    #     for i in range(N):
    #         if (self.bgmax[i][period] != 0 or self.bgmin[i][period] != 0):
    #             self.prosumers[i].utility[period] \
    #                 = 1 - ((self.bgmax[i][period] \
    #                         - self.prosumers[i].benefit[period])\
    #                         /(self.bgmax[i][period] - self.bgmin[i][period]))
            
    #         else:
    #             self.prosumers[i].utility[period] = 0
                
    # def computebenefit(self, period): 
    #     """
    #     Calculate benefit for each prosumer for a period
        
    #     Parameters
    #     ----------
    #     period: int 
    #         an instance of time t
        
    #     """
    #     N = self.prosumers.size
        
    #     for i in range(N):
    #         self.prosumers[i].benefit[period] \
    #             = self.prosumers[i].virtualben[period] \
    #                 + (self.czerom[period] \
    #                    * aux.apv(self.prosumers[i].consumption[period] \
    #                          - self.prosumers[i].production[period])
    #                    ) \
    #                 - self.prosumers[i].virtualcost[period]
                                                         
    # def uptdateprobaLRI(self, period, slowdown): 
    #     """
    #     Update probability for LRI based mode choice
        
    #     Parameters
    #     ----------
    #     period: int 
    #         an instance of time t
            
    #     slowdown: float
    #         Slowdown factor
            
    #     """
    #     N = self.prosumers.size
        
    #     for i in range(N):
    #         if self.prosumers[i].state[period] == ag.State.SURPLUS:
    #             if self.prosumers[i].mode[period] == ag.Mode.DIS :
    #                 self.prosumers[i].prmode[period][0] \
    #                     = min(1,
    #                           self.prosumers[i].prmode[period][0] \
    #                               + slowdown \
    #                                   * self.prosumers[i].utility[period] \
    #                                   * (1 - self.prosumers[i].prmode[period][0]))
    #                 self.prosumers[i].prmode[period][1] \
    #                     = 1 - self.prosumers[i].prmode[period][0]
                
    #             else :
    #                 self.prosumers[i].prmode[period][1] \
    #                     = min(1,
    #                           self.prosumers[i].prmode[period][1] \
    #                               + slowdown \
    #                                   * self.prosumers[i].utility[period] \
    #                                   * (1 - self.prosumers[i].prmode[period][1]))
    #                 self.prosumers[i].prmode[period][0] \
    #                     = 1 - self.prosumers[i].prmode[period][1]
                    
    #         elif self.prosumers[i].state[period] == ag.State.SELF:
    #             if self.prosumers[i].mode[period] == ag.Mode.DIS :
    #                 self.prosumers[i].prmode[period][0] \
    #                     = min(1,
    #                           self.prosumers[i].prmode[period][0] \
    #                               + slowdown \
    #                                   * self.prosumers[i].utility[period] \
    #                                   * (1 - self.prosumers[i].prmode[period][0]))
    #                 self.prosumers[i].prmode[period][1] \
    #                     = 1 - self.prosumers[i].prmode[period][0]
                
    #             else :
    #                 self.prosumers[i].prmode[period][1] \
    #                     = min(1,
    #                           self.prosumers[i].prmode[period][1] \
    #                               + slowdown \
    #                                   * self.prosumers[i].utility[period] \
    #                                   * (1 - self.prosumers[i].prmode[period][1]))
    #                 self.prosumers[i].prmode[period][0] \
    #                     = 1 - self.prosumers[i].prmode[period][1]
    #         else :
    #             if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS :
    #                 self.prosumers[i].prmode[period][0] \
    #                     = min(1,
    #                           self.prosumers[i].prmode[period][0] \
    #                               + slowdown \
    #                                   * self.prosumers[i].utility[period] \
    #                                   * (1 - self.prosumers[i].prmode[period][0]))
    #                 self.prosumers[i].prmode[period][1] \
    #                     = 1 - self.prosumers[i].prmode[period][0]
                
    #             else :
    #                 self.prosumers[i].prmode[period][1] \
    #                     = min(1,
    #                           self.prosumers[i].prmode[period][1] \
    #                               + slowdown \
    #                                   * self.prosumers[i].utility[period] \
    #                                   * (1 - self.prosumers[i].prmode[period][1]))
    #                 self.prosumers[i].prmode[period][0] \
    #                     = 1 - self.prosumers[i].prmode[period][1]
                     
    # def updatesmartgrid(self, period): 
    #     """
    #     Update storage for next period ie period+1, consit, prodit based on mode and state
        
    #     Parameters:
    #     ----------
    #     period : int
    #         an instance of time t
    #     """
    #     N = self.prosumers.size
        
    #     for i in range(N):
    #         if self.prosumers[i].state[period] == ag.State.DEFICIT:
    #             self.prosumers[i].prodit[period] = 0
    #             if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS:
    #                 self.prosumers[i].storage[period+1] = 0
    #                 self.prosumers[i].consit[period] \
    #                     = self.prosumers[i].consumption[period] \
    #                         - (self.prosumers[i].production[period] \
    #                            + self.prosumers[i].storage[period])
                
    #             else :
    #                 self.prosumers[i].storage[period+1] = self.prosumers[i].storage[period]
    #                 self.prosumers[i].consit[period] \
    #                     = self.prosumers[i].consumption[period] \
    #                         - self.prosumers[i].production[period]
            
    #         elif self.prosumers[i].state[period] == ag.State.SELF:
    #             self.prosumers[i].prodit[period] = 0
                
    #             if self.prosumers[i].mode[period] == ag.Mode.CONSMINUS:
    #                 self.prosumers[i].storage[period+1] = self.prosumers[i].storage[period]
    #                 self.prosumers[i].consit[period] \
    #                     = self.prosumers[i].consumption[period] \
    #                         - self.prosumers[i].production[period]
                
    #             else :
    #                 self.prosumers[i].storage[period+1] \
    #                     = self.prosumers[i].storage[period] \
    #                         - (self.prosumers[i].consumption[period] \
    #                            - self.prosumers[i].production[period])
    #                 self.prosumers[i].consit[period] = 0
    #         else :
    #             self.prosumers[i].consit[period] = 0
                
    #             if self.prosumers[i].mode[period] == ag.Mode.DIS:
    #                 self.prosumers[i].storage[period+1] \
    #                     = min(self.prosumers[i].smax,self.prosumers[i].storage[period] \
    #                           +(self.prosumers[i].production[period] \
    #                             - self.prosumers[i].consumption[period]))
    #                 self.prosumers[i].prodit[period] \
    #                     = aux.apv(self.prosumers[i].production[period] \
    #                           - self.prosumers[i].consumption[period] \
    #                               -(self.prosumers[i].smax \
    #                                 - self.prosumers[i].storage[period] ))
    #             else:
    #                 self.prosumers[i].storage[period+1] = self.prosumers[i].storage[period]
    #                 self.prosumers[i].prodit[period] \
    #                     = self.prosumers[i].production[period] \
    #                         - self.prosumers[i].consumption[period]
    
    # def updatemodeLRI(self, period, threshold): 
    #     """
    #     # Update mode using rules from LRI
        
    #     Parameters:
    #     ----------
    #     period : int
    #         an instance of time t
        
    #     threshold: float
    #         an threshold 
    #     """
    #     N = self.prosumers.size
        
    #     for i in range(N):
    #         rand = rdm.uniform(0,1)
            
    #         if self.prosumers[i].state[period] == ag.State.SURPLUS:
    #             if (rand <= self.prosumers[i].prmode[period][0] \
    #                 and self.prosumers[i].prmode[period][1] < threshold) \
    #                 or self.prosumers[i].prmode[period][0] > threshold :
    #                 self.prosumers[i].mode[period] = ag.Mode.DIS
                
    #             else :
    #                 self.prosumers[i].mode[period] = ag.Mode.PROD
            
    #         elif self.prosumers[i].state[period] == ag.State.SELF :
    #             if (rand <= self.prosumers[i].prmode[period][0] \
    #                 and self.prosumers[i].prmode[period][1] < threshold) \
    #                 or self.prosumers[i].prmode[period][0] > threshold :
    #                 self.prosumers[i].mode[period] = ag.Mode.DIS
                
    #             else :
    #                 self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
            
    #         else :
    #             if (rand <= self.prosumers[i].prmode[period][0] \
    #                 and self.prosumers[i].prmode[period][1] < threshold) \
    #                 or self.prosumers[i].prmode[period][0] > threshold :
    #                 self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
    #             else :
    #                 self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                    
    def updatemodeSDA(self, period):
        """
        Update mode using rules from SDA algorithm
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            sitminus = 0
            
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                sitminus = self.prosumers[i].storage[period]
                if self.prosumers[i].consumption[period+1] - self.prosumers[i].production[period+1] >= sitminus :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.PROD      
            
            elif self.prosumers[i].state[period] == ag.State.SELF:
                sitminus = self.prosumers[i].storage[period] - (self.prosumers[i].consumption[period] - self.prosumers[i].production[period])
                if self.prosumers[i].consumption[period+1] - self.prosumers[i].production[period+1] >= sitminus :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.DIS     
            else :
                sitminus = 0
                
                if self.prosumers[i].consumption[period+1] - self.prosumers[i].production[period+1] < sitminus :
                    self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                
    def updatemodeSyA(self, period): 
        """
        Update mode using rules from SyA algortihm
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
            
            else :
                self.prosumers[i].mode[period] = ag.Mode.DIS
             
    def updatemodeNosmart(self, period): 
        """
        Update mode using Nosmart algorithm
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                # Set values for X,Y,Z considering mode CONS+
                Xplus = 0
                Yplus = 0
                Zplus = self.prosumers[i].consumption[period] - (self.prosumers[i].production[period] + self.prosumers[i].storage[period])
                
                # Set values for X,Y,Z considering mode CONS-
                Xminus = self.prosumers[i].storage[period]
                Yminus = 0
                Zminus = self.prosumers[i].consumption[period] - self.prosumers[i].production[period]
                
                oitconsplus = aux.phiepominus(Zplus \
                                              + aux.apv(self.prosumers[i].consumption[period + 1] - \
                                                        (self.prosumers[i].production[period + 1] + Xplus))) \
                                - aux.phiepoplus(Yplus)
                oitconsminus = aux.phiepominus(Zminus \
                                           + aux.apv(self.prosumers[i].consumption[period + 1] - \
                                                      (self.prosumers[i].production[period + 1] + Xminus)))\
                                - aux.phiepoplus(Yminus)
                
                if oitconsplus > oitconsminus :
                    self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                    
            elif self.prosumers[i].state[period] == ag.State.SELF :
                
                # Set values for X,Y,Z considering mode DIS
                Xdis = self.prosumers[i].storage[period] \
                        - (self.prosumers[i].consumption[period] \
                               - self.prosumers[i].production[period])
                Ydis = 0
                Zdis = 0
                
                # Set values for X,Y,Z considering mode CONS-
                Xminus = self.prosumers[i].storage[period]
                Yminus = 0
                Zminus = self.prosumers[i].consumption[period] - self.prosumers[i].production[period]
                
                oitdis = aux.phiepominus(Zdis \
                                         + aux.apv(self.prosumers[i].consumption[period + 1] \
                                                   - (self.prosumers[i].production[period + 1] + Xdis))) \
                            - aux.phiepoplus(Ydis)
                oitconsminus = aux.phiepominus(Zminus \
                                               + aux.apv(self.prosumers[i].consumption[period + 1] 
                                                         - (self.prosumers[i].production[period + 1] + Xminus))) \
                                - aux.phiepoplus(Yminus)
                
                if oitdis > oitconsminus :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                
            else :
                
                # Set values for X,Y,Z considering mode PROD
                Xprod = self.prosumers[i].storage[period]
                Yprod = self.prosumers[i].production[period] - self.prosumers[i].consumption[period]
                Zprod = 0
                
                # Set values for X,Y,Z considering mode DIS
                Xdis = max(self.prosumers[i].smax,self.prosumers[i].storage[period] \
                           + (self.prosumers[i].production[period] \
                              - self.prosumers[i].consumption[period]))
                Ydis = self.prosumers[i].production[period] \
                        - self.prosumers[i].consumption[period] \
                        - (self.prosumers[i].smax - self.prosumers[i].storage[period])
                Zdis = 0

                oitdis = aux.phiepominus(Zdis + aux.apv(self.prosumers[i].consumption[period + 1] - \
                                                      (self.prosumers[i].production[period + 1] + Xdis))) \
                            - aux.phiepoplus(Ydis)
                oitprod = aux.phiepominus(Zprod + aux.apv(self.prosumers[i].consumption[period + 1] - \
                                                      (self.prosumers[i].production[period + 1] + Xprod))) \
                            - aux.phiepoplus(Yprod)
                
                if oitdis > oitprod :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.PROD
            
    # def updateState(self, period): 
    #     """
    #     Change prosumer's state based on its production, comsumption and available storage
        
    #     Parameters
    #     ----------
    #     period : int 
    #         an instance of time t
    #     """
    #     N = self.prosumers.size
        
    #     for i in range(N):    
    #         if self.prosumers[i].production[period] >= self.prosumers[i].consumption[period] :
    #             self.prosumers[i].state[period] = ag.State.SURPLUS
            
    #         elif self.prosumers[i].production[period] \
    #             + self.prosumers[i].storage[period] >= self.prosumers[i].consumption[period] :
    #             self.prosumers[i].state[period] = ag.State.SELF
            
    #         else :
    #             self.prosumers[i].state[period] = ag.State.DEFICIT
                
    # def updateMinmax(self, period, iteration):
    #     """
        

    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t
    #     iteration : int
    #         one step learning 

    #     Returns
    #     -------
    #     None.

    #     """
    #     N = self.prosumers.size

    #     for i in range(N):
    #         self.bgmin[i][period] \
    #             = min(self.prosumers[i].benefit[period], self.bgmin[i][period])
    #         self.bgmax[i][period] \
    #             = max(self.prosumers[i].benefit[period], self.bgmax[i][period])
            
            
    # The two following methods are used for production flexibility which is not yet totaly implemented
    
    def computeRealstate(self, period): 
        """
        Define real state based on real production and consumption

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
         
        N = self.prosumers.size
        
        for i in range(N):    
            if self.realprod[i][period] >= self.prosumers[i].consumption[period] :
                self.realstate[i][period] = ag.State.SURPLUS
            
            elif self.realprod[i][period] + self.prosumers[i].storage[period] >= self.prosumers[i].consumption[period] :
                self.realstate[i][period] = ag.State.SELF
            
            else :
                self.realstate[i][period] = ag.State.DEFICIT
    
    def computeRealmode(self, period): 
        """
        Change prosumer mode considering real state if needed 
        
        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.
        
        """
    
        N = self.prosumers.size
        
        for i in range(N):
            if (self.prosumers[i].state[period] == ag.State.SELF and self.realstate[i][period] == ag.State.DEFICIT) \
                or (self.prosumers[i].state[period] == ag.State.DEFICIT and self.realstate[i][period] == ag.State.SELF):
                self.prosumers[i].mode[period] == ag.Mode.CONSMINUS
            
            elif (self.prosumers[i].state[period] == ag.State.SELF and self.realstate[i][period] == ag.State.SURPLUS) \
                or (self.prosumers[i].state[period] == ag.State.SURPLUS and self.realstate[i][period] == ag.State.SELF):
                self.prosumers[i].mode[period] == ag.Mode.DIS
            
            elif self.prosumers[i].state[period] == ag.State.SURPLUS and self.realstate[i][period] == ag.State.DEFICIT:
                self.prosumers[i].mode[period] == ag.Mode.CONSMINUS
            
            elif self.prosumers[i].state[period] == ag.State.DEFICIT and self.realstate[i][period] == ag.State.SURPLUS:
                self.prosumers[i].mode[period] == ag.Mode.DIS
                


