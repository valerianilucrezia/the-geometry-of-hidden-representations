#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alessio ansuini (alessioansuini@gmail.com)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from math import sqrt
from scipy.spatial.distance import pdist, squareform
import os
import pandas as pd
import argparse
from argparse import Namespace
import random    
              
def estimate(X,fraction=0.8,verbose=False):    
    
    # sort distance matrix
    Y = np.sort(X,axis=1,kind='quicksort')
    # clean data
    k1 = Y[:,1]
    k2 = Y[:,2]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros) )
    good = np.setdiff1d(good,np.array(degeneracies))
    
    if verbose:
        print('Fraction good points: {}'.format(good.shape[0]/Y.shape[0]))
    
    k1 = k1[good]
    k2 = k2[good]    
    
    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0]*fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None,kind='quicksort')
    Femp = (np.arange(1,N+1,dtype=np.float64) )/N
    
    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])
   
    # work with na values 
    df = pd.DataFrame(x,columns = ['X'])
    df['Y'] = y

    df = df.dropna(axis= 0, how='any')
    
    x = df['X'].to_numpy()
    y = df['Y'].to_numpy()
    
     
    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[0:npoints,np.newaxis],y[0:npoints,np.newaxis]) 
    r,pval = pearsonr(x[0:npoints], y[0:npoints])  
    
    return x,y,regr.coef_[0][0],r,pval,npoints                    
  
                      
def block_analysis(X, resdir, name, blocks=list(range(1, 21)), fraction=0.8):
    #fitfile = open(resdir+'fit-res-'+name+".txt", "w")
    
    n = X.shape[0]
    dim = np.zeros(len(blocks))
    std = np.zeros(len(blocks))
    n_points = []
   
    for b in blocks:        
        idx = np.random.permutation(n)
        npoints = int(np.floor((n / b )))
        idx = idx[0:npoints*b]
        split = np.split(idx,b)      
        tdim = np.zeros(b)
        for i in range(b):
            I = np.meshgrid(split[i], split[i], indexing='ij')
            tX = X[tuple(I)]
            x,y,reg,r,pval,npoints = estimate(tX,fraction=fraction,verbose=False)
            
            if resdir != '':
                filename = resdir+name+'-fit'+str(b)+'-'+str(i)+'.png'
            
                fig = plt.figure(figsize=(6,6)) 
                plt.plot(x[0:npoints], y[0:npoints], '.', markersize = 7, alpha=0.3, label = 'datapoints')
                plt.plot(x[0:npoints], reg*x[0:npoints], label = 'linear fit')
            
                plt.legend()
                plt.tight_layout()
                plt.savefig(filename,dpi=200)
                plt.close()
   
            #fit for non-correlation -> plot x,y
            #fitfile.write('block:'+str(b)+', trial:'+str(i)+', pearson_coeff:'+str(r)+', pvalue:'+str(pval)+'\n')
            tdim[i] = reg          
        
        dim[blocks.index(b)] = np.mean(tdim)
        std[blocks.index(b)] = np.std(tdim)
        n_points.append(npoints)
    
    #fitfile.close()
    return dim,std,n_points



def save_ID_results(filename, dim, std, n_points):
    block_df = pd.DataFrame(columns =  ['dim','std', 'n_points'])
    block_df['dim'] = dim
    block_df['std'] = std
    block_df['n_points'] = n_points
    #block_df.to_csv(filename, sep = '\t', index = False)
    return block_df



def plot_ID_graph(filename, block_df, scaled, title = None):
    if title is None:
        title = 'ID_graph'
    
    fig = plt.figure(figsize=(10,6))
    plt.title(title)
    
    if scaled: 
        index = [i for i in range(len(block_df))]
        index.reverse()
        block_df['index'] = index

        plt.xticks(block_df['index'][::-1], labels = block_df['n_points'][::-1], rotation = 'vertical')
        plt.ylim(bottom= 0, top = max(block_df['dim'])+2)
        plt.errorbar(block_df['index'][::-1], block_df['dim'][::-1], block_df['std'][::-1])
    
    else:
        plt.xticks(block_df['n_points'][::-1], labels = block_df['n_points'][::-1], rotation = 'vertical')
        plt.ylim(bottom= 0, top = max(block_df['dim'])+2)
        plt.errorbar(block_df['n_points'][::-1], block_df['dim'][::-1], block_df['std'][::-1])
    
    plt.grid()
    plt.xlabel('N points')
    plt.ylabel('ID')
    plt.tight_layout()
    
    plt.savefig(filename,dpi=200)
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--inputdir", default='/u/area/lvaleriani/scratch/pnet-MSA/res/mean/pnet-rep-0.npy', type=str, help="input")
    parser.add_argument("-output", "--outputdir", default='', type=str, help="output")
      
    args = parser.parse_args()
    input = args.inputdir 
    output = args.outputdir
    
    random.seed(609) 
    
    name = input.split('/')[-1].split('-')[0]
    rep = ''.join(input.split('/')[-1].split('-')[1:3]).split('.')[0]
    print(rep,name)

    res = output + str(rep) + '/' 
    if not os.path.exists(res):
        os.makedirs(res,exist_ok = True)
    
    res_id = output + '/ids/'
    os.makedirs(res_id,exist_ok=True)
   
    d = input
    mean = np.load(d)
    print(mean.shape)
    print(mean,'\n\n\n')

    X = pdist(mean, 'euclidean')
    dist_mat = squareform(X)
        
    name_rep = name
    dim,std,n_point = block_analysis(dist_mat, res, name_rep, blocks=list(range(1, 21)), fraction=0.9)

    idfile = open(res_id +'id-'+name+'-'+str(rep)+".txt", "w")
    idfile.write('id-dimension: '+ str(dim) +'\n'+ 'mean-id-dimension: '+ str(sum(dim)/len(dim)) +
            '\n' +'std-dev: '+ str(std) + '\n'+'n_point: '+ str(n_point))
    idfile.close()

    block_df = save_ID_results(res+name+'-blockdf', dim, std, n_point)

    title = 'ID-'+name+'rep-'+str(rep)
    scaled = True
    plot_ID_graph(res+name+'-plot-ID-'+str(rep), block_df, scaled, title)
