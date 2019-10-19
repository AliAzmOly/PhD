#!/usr/bin/env python
# coding: utf-8

# In[918]:


from scipy import linalg as LA
from numpy  import *
import math

class CoupledHarmonicOscillator:

    def __init__(self, Step, Window, FreqIndex, Gain):
        
        assert(Step > 0), "Time step must be positive"
        assert(isinstance(Window, int) and Window > 1),"Window must be a scalar Int greater that unity"
        assert(isinstance(FreqIndex, ndarray) and FreqIndex.dtype == "int" and  all(FreqIndex >= 0)), "FreqIndex must be an array of positive Int"
        assert(FreqIndex.shape == (len(FreqIndex),1)), "FreqIndex must be a 2D array with shape.FreqIndex[1] = 1"
        assert(isinstance(Gain, ndarray)),"Gain must be an array"
        assert(Gain.shape[0]/Gain.shape[1] == len(FreqIndex)), "Gain and FreqIndex shape inconcitency"
       
        self.Step = Step
        self.FreqIndex = sort(FreqIndex, axis = 0)
        self.Fundamental_Freq = 2 * math.pi / (Window - 1) / Step
        self.Gain = Gain
            
         
    @property
    def Continuous(self):
        Input_matrix = self.ConjFlip(self.Gain)
        Frequencies = self.ConjFlip(1j*self.Fundamental_Freq*self.FreqIndex)
        State_matrix = kron(diagflat(Frequencies),identity(self.Dimension[1])) + tile(Input_matrix,(1,self.Dimension[0]))
        return (State_matrix, Input_matrix)

    @property    
    def Discrete(self):
        State_matrix = LA.expm(self.Continuous[0]*self.Step)  
        Input_matrix = (State_matrix - identity(self.Dimension[0]))@LA.inv(self.Continuous[0])@self.Continuous[1]
        return (State_matrix, Input_matrix)
    
    @property
    def Dimension(self):
        Length, Width = self.Gain.shape 
        Length = 2*Length - 1 if [0] in self.FreqIndex else 2*Length
        return(Length, Width)
    
    def UpdateOscillator(self, FrqGain_update):
        for FreqGain in FrqGain_update:
            assert(isinstance(FreqGain[0], ndarray) and FreqGain[0].dtype == "int" and  FreqGain[0] >= 0), "FreqIndex must be an array of positive Int"
            assert(isinstance(FreqGain[1], ndarray) or FreqGain[1] == None),"Gain must be an array or None"
            assert(FreqGain[1].shape == (self.Dimension[1],self.Dimension[1])), "Gain shape inconcitency"
            UpdatingIndex = self.UpdateIndex(FreqGain[0])
            if FreqGain[0] not in self.FreqIndex:# Add elemengts
                self.FreqIndex = insert(self.FreqIndex,UpdatingIndex,FreqGain[0],axis = 0)
                Add_index = (UpdatingIndex - 1)*self.Dimension[1] + 1
                self.Gain = insert(self.Gain, Add_index, FreqGain[1],axis = 0)
            elif FreqGain[1] == None:# Remove elemenst
                self.FreqIndex = delete(self.FreqIndex,UpdatingIndex,axis = 0)
                Remove_index = range(UpdatingIndex *self.Dimension[1],(UpdatingIndex + 1)*self.Dimension[1])
                self.Gain = delete(self.Gain, Remove_index, axis = 0)
            else:#Replace elements
                Replace_index = range(UpdatingIndex*self.Dimension[1],(UpdatingIndex + 1)*self.Dimension[1])
                self.Gain[Replace_index,:] = FreqGain[1]    
    
    def UpdateIndex(self,X_index):
        if X_index in self.FreqIndex:
            return where(self.FreqIndex == X_index)[0][0]
        return self.FreqIndex.size if X_index > max(self.FreqIndex) else argmax(self.FreqIndex > X_index) 
   
    def ConjFlip(self, X_array):
        Length, Width = X_array.shape
        Flip_Operator = flip(identity(Length), axis = 0)
        if [0] in self.FreqIndex:
            Flip_Operator = delete(Flip_Operator, range(Length - Width, Length), axis = 0)
        return vstack([conj(Flip_Operator @ X_array),X_array]) 

