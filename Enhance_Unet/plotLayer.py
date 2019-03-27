import numpy as np
#import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def getXY(A):
    xind = range(A.shape[0])
    yind = range(A.shape[1])
    xx = []
    yy = []
    for p in zip(xind, A):
        for i in yind:
            if p[1][i] > 0:
                xx.append(p[0])
                yy.append(i)
    return xx, yy

def plotXY(X, Y):
    plt.scatter(X, Y)
    plt.show()

def plotLayers(hL, show=False):
    for p in hL:
        plt.figure()
        z, phi = getXY(p)
        plt.scatter(phi, z)
        plt.xlim(0, p.shape[1])
        plt.ylim(0, p.shape[0])

    if show:
        for i in plt.get_fignums():
            plt.show(i)

def plotLayersSinglePlot(hL, show=False):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    bar = ['lightblue', 'skyblue','deepskyblue', 'royalblue', 'blue', 'navy']
    plt.figure()
    for i in range(len(hL)):
        p = hL[i]
        z, phi = getXY(p)
        plt.scatter(phi, z, color=colors[bar[i]], label='HL' + str(i))
        plt.xlim(0, p.shape[1])
        plt.ylim(0, p.shape[0])

    if show:
        for i in plt.get_fignums():
            plt.legend()
            plt.show(block=True)

def PlotImage(hL, gthL, title1="Input Image", title2="Ground Truth Image", show=False):
    plotLayersSinglePlot(hL, show=show)
    plt.title(title1)
    plt.legend()
    plotLayersSinglePlot(gthL, show=show)
    plt.title(title2)
    plt.legend()
