"""
@author: alessio ansuini (alessioansuini@gmail.com)
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
import pandas as pd

import plotly.graph_objects as go

              
def estimate(Y, fraction=0.8, verbose=False):  
    # print('start') 
    k1 = Y[:,0]
    k2 = Y[:,1]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(k1.shape[0]), np.array(zeros) )
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
    r, pval = pearsonr(x[0:npoints], y[0:npoints])  
    
    return x, y, regr.coef_[0][0], r, pval, npoints                    
  
                      
def block_analysis(X,blocks=list(range(1, 21)), fraction=0.8):    
    dim = np.zeros(len(blocks))
    std = np.zeros(len(blocks))
    n_points = []
   
    for b in blocks:        
        # print(b)
        tdim = np.zeros(b)
        
        for i in range(b):
            load_file = np.load(X+'_block-'+str(b)+'-'+str(i)+'.npy')
            _, _, reg, _, _, npoints = estimate(load_file,fraction=fraction, verbose=False)
            tdim[i] = reg            
       
        dim[blocks.index(b)] = np.mean(tdim)
        std[blocks.index(b)] = np.std(tdim)
        n_points.append(npoints)
    
    return dim, std, n_points


# def save_ID_results(dim, std, n_points, filename='', save=False):
#     block_df = pd.DataFrame(columns =  ['dim','std','n_points'])
#     block_df['dim'] = dim
#     block_df['std'] = std
#     block_df['n_points'] = n_points
    
#     if save:
#         if filename != '':
#             block_df.to_csv(filename, sep='\t', index=False)
#         else:
#             'Missing filename!'
    
#     return block_df


def plot_ID(fig, reps, mean, cline, name, r=1, c=1, legend=True):
    fig.add_trace(go.Scatter(x = reps/(len(reps)-1),
                            y = mean,
                            mode ='lines+markers',
                            name = name,
                            marker = dict(color=cline, size=9),
                            line = dict(color=cline, width=3),
                            showlegend = legend,      
                            ),
                    row = r, 
                    col = c
                )

    fig.update_xaxes(showline = True, 
                    linewidth = 1, 
                    linecolor = 'black',
                    showticklabels = True, 
                    tickmode = 'array',
                    tickvals = np.arange(0, 1.1, 0.1),
                    range = [-0.01,1.02],
                    title = 'relative depth',
                    ticks = 'outside'
                    )

    fig.update_yaxes(showline = True, 
                    linewidth = 1, 
                    linecolor = 'black',
                    showticklabels = True, 
                    tickmode = 'array',
                    tickvals = [i for i in range(0,26,2)],
                    title = 'ID',
                    range = [0,24],
                    ticks = 'outside'
                    )
    return fig





