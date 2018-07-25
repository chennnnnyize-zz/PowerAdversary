import numpy as np
import csv
import matplotlib.pyplot as plt


def voltage_sag(signal, level):
    x=np.linspace(4*np.pi, 10*np.pi, 300)
    y=level*np.sin(x)
    signal[200:500]=y
    return signal

def voltage_distortion(signal, level):
    noise=np.random.normal(loc=0, scale=level, size=np.shape(signal))
    signal+=noise
    #plt.plot(signal)
    #plt.show()

    return signal


def voltage_impulse(signal, level):
    noise=np.random.normal(loc=0, scale=level, size=np.shape(signal))
    signal[400:420]+=noise[400:420]
    #plt.plot(signal)
    #plt.show()

    return signal




if __name__ == '__main__':
    x = np.linspace(0 * np.pi, 20* np.pi, 1000)
    y = np.sin(x)
    signal=y
    signal_all=[]
    '''for i in range(200):
        levels=np.random.uniform(0.5, 0.9)
        #print(levels)
        signals=voltage_sag(signal, level=levels)
        #plt.plot(signals)
        #plt.show()
        signal_all.append(np.copy(signals))
    signal_all=np.array(signal_all, dtype=float).reshape(-1,1000)
    with open('sag.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(signal_all)'''


    '''signal_all=[]
    for i in range(200):
        levels=np.random.uniform(0.0, 0.1)
        #print(levels)
        signals=voltage_distortion(signal, level=levels)
        #plt.plot(signals)
        #plt.show()
        signal_all.append(np.copy(signals))
    signal_all=np.array(signal_all, dtype=float).reshape(-1,1000)
    with open('distortion.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(signal_all)'''


    '''signal_all=[]
    for i in range(200):
        levels=np.random.uniform(0.5, 0.8)
        #print(levels)
        signals=voltage_impulse(signal, level=levels)
        #plt.plot(signals)
        #plt.show()
        signal_all.append(np.copy(signals))
    signal_all=np.array(signal_all, dtype=float).reshape(-1,1000)
    with open('impulse.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(signal_all)'''

    signal_all = []
    for i in range(200):
        levels = np.random.uniform(0.0, 0.01)
        # print(levels)
        signals = voltage_distortion(signal, level=levels)
        # plt.plot(signals)
        # plt.show()
        signal_all.append(np.copy(signals))
    signal_all = np.array(signal_all, dtype=float).reshape(-1, 1000)
    with open('normal.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(signal_all)


