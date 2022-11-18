import sys
sys.path.insert(0,'/u/area/lvaleriani/scratch/hier-nucl/hierarchical_nucleation/')
import argparse
import numpy as np
import os
import faiss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--inputdir", default='', type=str, help="input")
    parser.add_argument("-output", "--outputdir", default='', type=str, help="output")
    parser.add_argument("-kneigh", "--neigh", default=40, type=int, help="k-neigh")

    args = parser.parse_args()
    input = args.inputdir
    output = args.outputdir
    k = args.neigh
    
    all_data = np.load(input)

    rep = 'rep'+input.split('/')[-1].split('-')[2].split('.')[0]

    index = faiss.IndexFlatL2(all_data.shape[1])
    all_data = all_data.astype('float32')
    index.add(all_data)
    neigh, idx_neig = index.search(all_data, k)
    np.save(output+rep+'-neigh.npy',neigh)
    np.save(output+rep+'-idx.npy',idx_neig)
