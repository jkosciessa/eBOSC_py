"""
BOSC (Better Oscillation Detection) function library
Rewritten from MATLAB to Python by Julian Q. Kosciessa

The original license information follows:
---
This file is part of the Better OSCillation detection (BOSC) library.

The BOSC library is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The BOSC library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2010 Jeremy B. Caplan, Adam M. Hughes, Tara A. Whitten
and Clayton T. Dickson.
---
"""

import numpy as np

def BOSC_tf(eegsignal,F,Fsample,wavenumber):

    st=1./(2*np.pi*(F/wavenumber))
    A=1./np.sqrt(st*np.sqrt(np.pi))
    # initialize the time-frequency matrix
    B = np.zeros((len(F),len(eegsignal)))
    B[:] = np.nan
    # loop through sampled frequencies
    for f in range(len(F)):
        #print(f)
        t=np.arange(-3.6*st[f],(3.6*st[f]),1/Fsample)
        # define Morlet wavelet
        m=A[f]*np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*F[f]*t)
        y=np.convolve(eegsignal,m, 'full')
        y=abs(y)**2
        B[f,:]=y[np.arange(int(np.ceil(len(m)/2))-1, len(y)-int(np.floor(len(m)/2)), 1)]
        T=np.arange(1,len(eegsignal)+1,1)/Fsample
    return B, T, F


def BOSC_detect(b,powthresh,durthresh,Fsample):
    """
    detected=BOSC_detect(b,powthresh,durthresh,Fsample)
    This function detects oscillations based on a wavelet power
    timecourse, b, a power threshold (powthresh) and duration
    threshold (durthresh) returned from BOSC_thresholds.m.
    
    It now returns the detected vector which is already episode-detected.
    
    b - the power timecourse (at one frequency of interest)
    
    durthresh - duration threshold in  required to be deemed oscillatory
    powthresh - power threshold
    
    returns:
    detected - a binary vector containing the value 1 for times at
               which oscillations (at the frequency of interest) were
               detected and 0 where no oscillations were detected.
    
    note: Remember to account for edge effects by including
    "shoulder" data and accounting for it afterwards!
    
    To calculate Pepisode:
    Pepisode=length(find(detected))/(length(detected));
    """                           

    # number of time points
    nT=len(b)
    #t=np.arange(1,nT+1,1)/Fsample
    
    # Step 1: power threshold
    x=b>powthresh
    # we have to turn the boolean to numeric
    x = np.array(list(map(int, x)))
    # show the +1 and -1 edges
    dx=np.diff(x)
    if np.size(np.where(dx==1))!=0:
        pos=np.where(dx==1)[0]+1
        #pos = pos[0]
    else: pos = []
    if np.size(np.where(dx==-1))!=0:
        neg=np.where(dx==-1)[0]+1
        #neg = neg[0]
    else: neg = []

    # now do all the special cases to handle the edges
    detected=np.zeros(b.shape)
    if not any(pos) and not any(neg):
        # either all time points are rhythmic or none
        if all(x==1):
            H = np.array([[0],[nT]])
        elif all(x==0):
            H = np.array([])
    elif not any(pos):
        # i.e., starts on an episode, then stops
        H = np.array([[0],neg])
        #np.concatenate(([1],neg), axis=0)
    elif not any(neg):
        # starts, then ends on an ep.
        H = np.array([pos,[nT]])
        #np.concatenate((pos,[nT]), axis=0)
    else:
        # special-case, create the H double-vector
        if pos[0]>neg[0]:
            # we start with an episode
            pos = np.append(0,pos)
        if neg[-1]<pos[-1]:
            # we end with an episode
            neg = np.append(neg,nT)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        H = np.array([pos,neg])
        #np.concatenate((pos,neg), axis=0)
    
    if H.shape[0]>0: 
        # more than one "hole"
        # find epochs lasting longer than minNcycles*period
        goodep=H[1,]-H[0,]>=durthresh
        if not any(goodep):
            H = [] 
        else: 
            H = H[:,goodep.nonzero()][:,0]
            # mark detected episode on the detected vector
            for h in range(H.shape[1]):
                detected[np.arange(H[0][h], H[1][h],1)]=1
        
    # ensure that outputs are integer
    detected = np.array(list(map(int, detected)))
    return detected