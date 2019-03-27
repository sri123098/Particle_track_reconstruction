from __future__ import division
from parseHit import *
from plotLayer import *

import pickle
from scipy import sparse
from scipy.ndimage import zoom
from skimage.measure import block_reduce
import time

# for remote display
import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
# # #


def CylinderPadding(hL, row):
    return [np.append(np.append(L[:, -row:], L, axis=1), L[:, 0:row], axis=1) for L in hL]

def TopBotPadding(hL, row):
    zeros = np.zeros((row, hL[0].shape[1]))
    return [np.append(np.append(zeros, L, axis=0), zeros, axis=0) for L in hL]

def PlotImage(hL, gthL, title1="Input Image", title2="Ground Truth Image", show=False):
    plotLayersSinglePlot(hL, show=show)
    plt.title(title1)
    plt.legend()
    plotLayersSinglePlot(gthL, show=show)
    plt.title(title2)
    plt.legend()

def Preprocess(dataset_name, save_file="", plot=False):
    with open(dataset_name, "rb") as f:
        events = pickle.load(f)

    print("Preprocessing training data from %s" %dataset_name)
    start = time.time()

    dataset = {}
    x = []
    y = []
    for event in events:
        print("processing %d/%d time start:%f secs" %(len(x) + 1, len(events), time.time() - start))
        hL = [a.toarray() for a in event.hL]
        gthL = [a.toarray() for a in event.gthL]


        #start_pool = time.time()

        # do max pooling to resize image
#        hL = [block_reduce(a, block_size=(2,2), func=np.max) for a in hL]
#        hL = [block_reduce(a, block_size=(2,2), func=np.max) for a in hL]
#        hL = [block_reduce(a, block_size=(2,2), func=np.max) for a in hL]
#        hL = [block_reduce(a, block_size=(2,1), func=np.max) for a in hL]
#
#        gthL = [block_reduce(a, block_size=(2,2), func=np.max) for a in gthL]
#        gthL = [block_reduce(a, block_size=(2,2), func=np.max) for a in gthL]
#        gthL = [block_reduce(a, block_size=(2,2), func=np.max) for a in gthL]
#        gthL = [block_reduce(a, block_size=(2,1), func=np.max) for a in gthL]

#        print("Max pooling array %f" %(time.time() - start_pool))

        # manually add padding rows and columns, pad size = filter size / 2
        hL = CylinderPadding(hL, 3)
        hL = TopBotPadding(hL, 3)
        gthL = CylinderPadding(gthL, 3)
        gthL = TopBotPadding(gthL, 3)
        x.append(np.array(hL))
        y.append(np.array(gthL))

        if plot:
            PlotImage(hL, gthL, show=False)
            plt.show(block=True)

    dataset["x"] = x
    dataset["y"] = y

    print("%d events proceessed in %f secs. x.shape:%s y.shape:%s" %(len(y), time.time() - start, str(x[0].shape), str(y[0].shape) ))
    if save_file != "":
        print("Save file to %s" %(save_file))
        with open(save_file, "wb") as f:
            pickle.dump(dataset, f)

    return dataset

if __name__ == "__main__":
    #file_path = "ExampleData/"
    file_path = "Data/"
    dataset_name = file_path + "d0_sparse_dataset.dat"
    #save_file_name = file_path + "test_proc_data.dat"
    save_file_name = ""

    dataset = Preprocess(dataset_name, save_file_name, plot=True)

    #with open(saved_file_name, 'wb') as f:
        #pickle.dump(dataset, f)



