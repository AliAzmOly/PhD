#!/usr/bin/env python
# coding: utf-8

# In[806]:


from scipy import linalg as LA
from numpy  import *
import math

class CoupledHarmonicOscillator:

    def __init__(self, Window, FreqIndex, Gain):
        
        assert(isinstance(Window, int) and Window > 1),"Window must be a scalar Int greater that unity"
        assert(isinstance(FreqIndex, ndarray) and FreqIndex.dtype == "int" and  all(FreqIndex >= 0)), "FreqIndex must be an array of positive Int"
        assert(FreqIndex.shape == (len(FreqIndex),1)), "FreqIndex must be a 2D array with shape.FreqIndex[1] = 1"
        assert(isinstance(Gain, ndarray)),"Gain must be an array"
        assert(Gain.shape[0]/Gain.shape[1] == len(FreqIndex)), "Gain and FreqIndex shape inconcitency"
       
        self.FreqIndex = sort(FreqIndex, axis = 0)
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
    
    def UpdateFilter(self, FrqGain_List):
        for FreqGain in FrqGain_List:
            UpdatingIndex = self.UpdateIndex(FreqGain[0])
            Dim = self.Gain.shape[1]
            if FreqGain[0] not in self.FreqIndex:#Add
                self.FreqIndex = insert(self.FreqIndex,UpdatingIndex,FreqGain[0],axis = 0)
                self.Gain = insert(self.Gain,(UpdatingIndex - 1)*Dim + 1,FreqGain[1],axis = 0)
            elif FreqGain[1] == None:#Remove
                self.FreqIndex = delete(self.FreqIndex,UpdatingIndex,axis = 0)
                self.Gain = delete(self.Gain, slice(UpdatingIndex *Dim,(UpdatingIndex + 1)*Dim), axis = 0)
            else:#Replace
                self.Gain[UpdatingIndex*Dim:(UpdatingIndex + 1)*Dim,:] = FreqGain[1]    
    
    def UpdateIndex(self,X_index):
        if X_index in self.FreqIndex:
            return where(self.FreqIndex == X_index)[0][0]
        return self.FreqIndex.size if X_index > max(self.FreqIndex) else argmax(self.FreqIndex > X_index) 
   
    def ConjFlip(self, X_array):
        Width, Dim = X_array.shape
        Flip_Operator = flip(identity(Width), axis = 0)
        if [0] in self.FreqIndex:
            Flip_Operator = delete(Flip_Operator, range(Width - Dim, Width), axis = 0)
        return vstack([conj(Flip_Operator @ X_array),X_array]) 

    
    

    


# In[807]:


H =CoupledHarmonicOscillator(2, array([0,1]), array([[0.5],[0.5]]))


# In[778]:


Freq = H.FreqIndex


# In[704]:


Freq.dtype


# In[705]:


Freq


# In[684]:


Freq.dtype == "int"


# In[680]:


print(Freq < 0)


# In[683]:


if not all(Freq >= 0):
    print("negative value found")


# In[682]:


S = Freq <= 0
print(S)


# In[716]:


S = [[2],[3]]


# In[717]:


S.dtype


# In[ ]:




