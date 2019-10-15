#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import linalg as LA
from numpy  import *
import math

class HO_Filter:

    def __init__(self, FreqIndex, Window, Gain):
        
        FreqIndex.shape = (len(FreqIndex),1)
        self.FreqIndex = FreqIndex
        Width, Dim = Gain.shape
        Gain.shape = (max(Width,Dim), min(Width,Dim))
        try:
            if max(Width,Dim)/min(Width,Dim) != len(FreqIndex):
                raise Exception 
        except Exception as e:
            print("Gain and Frequency Index are not Consistent")
        self.Fundamental_Freq = 2 * math.pi / (Window - 1)
        self.Gain = Gain
         
    @property
    def Oscillator(self):
        InputMatrix = self.ConjFlip(self.Gain)
        Width, Dim = InputMatrix.shape
        FrequencyVector = self.ConjFlip(1j*self.Fundamental_Freq*self.FreqIndex)
        StateMatrix = kron(diagflat(FrequencyVector),identity(Dim)) + tile(InputMatrix,(1,Width))
        return (StateMatrix, InputMatrix)

    @property    
    def Filter(self):
        Width = len(self.Oscillator[1])
        StateMatrix = LA.expm(self.Oscillator[0])  
        InputMatrix = (StateMatrix - identity(Width))@LA.inv(self.Oscillator[0])@self.Oscillator[1]
        return (StateMatrix, InputMatrix) 
   
    def ConjFlip(self, X_array):
        Width, Dim = X_array.shape
        Flip_Operator = flip(identity(Width), axis = 0)
        if [0] in self.FreqIndex:
            Flip_Operator = delete(Flip_Operator, range(Width - Dim, Width), axis = 0)
        return vstack([conj(Flip_Operator @ X_array),X_array]) 

    
    

    

