import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, dct, idct
from scipy.io import wavfile # get the api
import numpy as np
import math

fs, audio = wavfile.read('C:/Users/win/Desktop/0.1_1.wav')

audio = audio.T[0]
samples = audio.shape[0]
L = (samples / fs)*1000


f, ax = plt.subplots()
ax.plot((np.arange(samples) / fs)*1000, audio)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Amplitude')


def Edft(x,L): #calculating error dft for x
    y = fft(x)
    N = len(y)
    a = int((N+1-L)/2)
    b = int((N-1+L)/2)
    for i in range(a,b+1):
        y[i] = 0
    x_m = ifft(y)

    return ((x - x_m) ** 2).mean(axis=0)

def Edct(x,L): #calculating error dct for x
    y = dct(x)
    N = len(y)
    for i in range(N-L,N):
        y[i] = 0
    x_m = idct(y)/(2*len(x))

    return ((x - x_m) ** 2).mean(axis=0)  

h2 = np.array([[1,1],[1,-1]])
def haar_mat(n):
    n = int(n)
    if n == 1:
        return h2
    else:
        a = np.kron(haar_mat(n-1),[1,1])
        b = np.kron(np.identity(int(math.pow(2,n-1)))*math.pow(2,(n-1)/2.0),[1,-1])
        #print(np.concatenate((a,b),axis=0))
        return np.concatenate((a,b),axis=0)

def haar(x):
    return np.matmul(haar_mat(math.log(len(x),2)),np.transpose(x))

def ihaar(y): #calculating inverse haar for x
    n = int(math.log(len(y),2))
    N = len(y)
    hn = haar_mat(n)
    return np.matmul(np.transpose(hn)/N,np.transpose(y))

def Ehaar(x,L): #calculating error for haar transform
    y = haar(x)
    N = len(y)
    for i in range(N-L,N):
        y[i] = 0
    x_m = ihaar(y)

    return ((x - x_m) ** 2).mean(axis=0)  



n = len(audio)
dft = fft(audio).real

plt.plot(dft)
plt.title('Complete DFT plot')
plt.show()

dct_audio = dct(audio)

plt.plot(dct_audio)
plt.title('Complete DCT plot')
plt.show()

haar_audio = haar(audio[0:4096])

plt.plot(haar_audio)
plt.title('Complete Haar plot')
plt.show()

edft = [0.]*len(audio)
edct = [0.]*len(audio)
ehaar = [0.]*len(audio)

for L in range(len(audio)):
    edft[L] = Edft(audio,L)
    edct[L] = Edct(audio,L)
    ehaar[L] = Edct(audio[0:4096],L)
    print(L)
    
ehaar1 = np.array(ehaar)    
#rescaling factor
ehaar1 = ehaar1 *1700 
# Comparision
fig, ax = plt.subplots()
ax.plot(edft, 'k:', label='DFT') 
ax.plot(edct,'k--', label = 'DCT') 
ax.plot(ehaar1, 'k', label='HAAR')
ax.grid()
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')    