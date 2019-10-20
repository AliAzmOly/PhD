#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import linalg as LA
from numpy  import *
import math
import itertools

class TimeSeries(HarmonicOscillators):
    
    def __init__(self, Signal, Step):
        Window = Signal.size
        FreqIndex = arange(Window).reshape(Window,1)
        Gain = 2/Step/(Window - 1)*ones((Window,1))
        super().__init__(Step, Window, FreqIndex, Gain)
        self.Spectrum = self.Harmonics(Signal - Signal[0,0], zeros((2*Window - 1,1)), Window)[:,-1]
        self.Signal = Signal
    
    def CleanSpectrum(self, FiltIndex, Type):
        if Type == "Stop":
            FiltIndex = delete(self.FreqIndex, FiltIndex, axis = 0)
        FreqGain_update = list(itertools.zip_longest(list(delete(self.FreqIndex, FiltIndex, axis = 0)), [None]))
        self.UpdateOscillator(FreqGain_update)
        CleanIndex = (self.Signal.size - 1) + FiltIndex 
        return(self.ConjFlip(self.Spectrum[CleanIndex]))
        
    def CleanSignal(self, FiltIndex, Type):
        CleanSpectrum = self.CleanSpectrum(FiltIndex, Type)
        ExtendedSignal = (self.Signal[0,-1] - self.Signal[0,0])*ones((1,self.Signal.size))
        ExtendedHarmonics = self.Harmonics(ExtendedSignal, CleanSpectrum, self.Signal.size)
        return(2*sum(ExtendedHarmonics, axis = 0).real - ExtendedSignal + self.Signal[0,0])

