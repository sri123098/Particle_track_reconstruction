from __future__ import division
import json
import numpy as np
import pickle
import time
import glob
import os
import bz2
from pprint import pprint
from scipy import sparse
from scipy.ndimage import zoom
from pandas.io.json import json_normalize


class Data:
    def __init__(self, hL, gthL, label, event_id):

        self.hL = hL
        self.gthL = gthL
        self.label = label
        self.id = event_id

    def __str__(self):
        return str(self.id) + " " + str(self.label)

def ValidTrack(track, id_map, layer_threshold=3):
    valid = [0] * 6
    for hit_id in track:
        hit = id_map[hit_id]
        ind = hit["Layer"] * 2 + hit["HalfLayer"]
        valid[ind] = 1

    return sum(valid) >= layer_threshold

def ParseHits(json_file, file_type=True, save_file="", log=True):
    if file_type:
        with open(json_file, "r") as f:
            raw_data = json.load(f)
    else:
        raw_data = json.loads(json_file)

    # prepare matrices index
    ZIndex_max = raw_data["Events"][0]["MetaData"]["Layer0"]["PixelZIndex_Count"]
    L0Phi_max = raw_data["Events"][0]["MetaData"]["Layer0"]["PixelPhiIndexInLayer_Count"]
    L1Phi_max = raw_data["Events"][0]["MetaData"]["Layer1"]["PixelPhiIndexInLayer_Count"]
    L2Phi_max = raw_data["Events"][0]["MetaData"]["Layer2"]["PixelPhiIndexInLayer_Count"]
    #Phi = L0Phi_max
    Phi = 1024
    Z = 1024
    Phi_ratio = [Phi / L0Phi_max, Phi / L1Phi_max, Phi / L2Phi_max]
    Z_ratio = Z / ZIndex_max

    dataset = []

    if file_type:
        print("start to extract from %s." %json_file)
    count = 0
    start = time.time()
    for event in raw_data["Events"]:
        count += 1
        if log:
            print("Parsing event %d/%d time start:%f secs" %(count, len(raw_data["Events"]), time.time() - start))
        label = [1 if event["TruthTriggerFlag"]["Flags"][flag] else 0 for flag in event["TruthTriggerFlag"]["Flags"] ]
        event_id = event["MetaData"]["EventID"]
        id_map = {}

        hL = [sparse.lil_matrix((Z,  Phi)) for i in range(6)]
        gthL = [sparse.lil_matrix((Z,  Phi)) for i in range(6)]

        for hit in event["RawHit"]["MVTXHits"]:
            # Store hit according to id
            id_map[hit["ID"]["HitSequenceInEvent"]] = hit["ID"]

            hL_ind = hit["ID"]["Layer"] * 2 + hit["ID"]["HalfLayer"]
            hL[hL_ind][int(hit["ID"]["PixelZIndex"] * Z_ratio), int(hit["ID"]["PixelPhiIndexInLayer"] * Phi_ratio[hit["ID"]["Layer"]])] = 1


        for track in event["TruthHit"]["TruthTracks"]:
            if ValidTrack(track["HitSequenceInEvent"], id_map, 3):
                for hit_id in track["HitSequenceInEvent"]:
                    hit = id_map[hit_id]

                    gthL_ind = hit["Layer"] * 2 + hit["HalfLayer"]
                    gthL[gthL_ind][int(hit["PixelZIndex"] * Z_ratio), int(hit["PixelPhiIndexInLayer"] * Phi_ratio[hit["Layer"]])] = 1


        d = Data(hL, gthL, label, event_id)

        dataset.append(d)

    if save_file != "":
        with open(save_file, "wb") as f:
            pickle.dump(dataset, f)
    if log:
        if file_type:
            print("Extract from %s finished. %d events fetched in %f secs."
                %(json_file, len(dataset), time.time() - start))
        else:
            print("Extract finished. %d events fetched in %f secs."
                %(len(dataset), time.time() - start))

    return dataset



if __name__ == "__main__":

    folder = "Data/"
    save_folder = "1stDataset/"
    if not os.path.exists(folder + save_folder):
        os.makedirs(folder + save_folder)

    print("Parsing %s" %(folder))

    D0_dir = glob.glob(folder + "D0*/*.bz2")
    #D0_dir = glob.glob(folder + "SmallD0*/*.bz2")
    False_dir = glob.glob(folder + "Inclusive*/*.bz2")
    #False_dir = glob.glob(folder + "SmallInclusive*/*.bz2")

    d0_save_file = folder + save_folder + "d0_sparse_dataset"
    false_save_file = folder + save_folder + "false_sparse_dataset"

    count = 0
    start = time.time()

    for zip_file in D0_dir:
        dataset = []
        count += 1
        print("Parsing from %s %d/%d time elapsed:%f secs" %(os.path.basename(zip_file), count, len(D0_dir), time.time() - start))
        with open(zip_file) as z:
            json_file = bz2.decompress(z.read())
            dataset = ParseHits(json_file, file_type=False, log=False)

        save_name = d0_save_file + "%04d" %count + ".dat"
        print("Writing to %s" %(save_name))
        with open(save_name, "wb") as f:
            pickle.dump(dataset, f)
        print("Writing to %s finished." %(d0_save_file))


    count = 0
    start = time.time()

    for zip_file in False_dir:
        dataset = []
        count += 1
        print("Parsing from %s %d/%d time elapsed:%f secs" %(os.path.basename(zip_file), count, len(D0_dir), time.time() - start))
        with open(zip_file) as z:
            json_file = bz2.decompress(z.read())
            dataset = ParseHits(json_file, file_type=False, log=False)

        save_name = false_save_file + "%04d" %count + ".dat"
        print("Writing to %s" %(save_name))
        with open(save_name, "wb") as f:
            pickle.dump(dataset, f)
        print("Writing to %s finished." %(false_save_file))
