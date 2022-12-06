import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def get_data(path, nlayer, label_path='', ng=100):
    ds = create_ds(nlayer)
    for i in np.arange(0, nlayer+1):

        neig = np.load(path + str(i) + '-neigh.npy')
        idx = np.load(path + str(i) + '-idx.npy')

        if label_path != '':
            label = open(label_path)
            label = np.array([l.strip('\n') for l in label])
            ds = add_data(ds, i, neig, idx, ng, label)
        
        else:
            ds = add_data(ds, i, neig, idx, ng)

    return ds


def create_ds(nlayer):
    ds = {i : {} for i in range(nlayer+1)}
    return ds


def add_data(ds, layer, distances, idxs, knn, labels = None):
    ds[layer]['distances'] = distances[:, :knn+1]
    ds[layer]['idxs'] = idxs[:, :knn+1]

    if type(labels) != str:
        ds[layer]['labels'] = labels
    return ds


def overlap_label(ds, k):
    overlap_alls = []

    for l in ds.keys():
        overlap = 0.

        idxs = ds[l]['idxs'] 
        nelem = idxs.shape[0]
        gt_labels = ds[l]['labels']

        for i in range(nelem):
            neigh_idx_i = idxs[i, 1:k + 1]
            overlap += sum(gt_labels[neigh_idx_i] == gt_labels[i]) / k
        overlap = overlap / nelem
        overlap_alls.append(overlap)

    return overlap_alls


def overlap_layer(ds, l1, l2, k):
    idxs_1 = ds[l1]['idxs'] 
    idxs_2 = ds[l2]['idxs'] 

    nelem = idxs_1.shape[0]
    ov = []
    for i in range(nelem):
        ov.append(len(set(idxs_1[i, 1:k + 1]) & set(idxs_2[i, 1:k + 1])))
    return np.mean(ov) /k


def plot_no(fig, layer, df, color, lab, title, row=1, col=1, legend=True):
    num = np.arange(layer+1)
    fig.add_trace(go.Scatter(x = num/(len(num)-1),
                            y = df,
                            mode = 'lines + markers',
                            name = lab,
                            marker = dict(color=color,size=8),
                            line = dict(color=color,width=1.5),
                            showlegend = legend
                        ), row = row, col = col,
                )
    fig.update_xaxes(showline = True, 
                    linewidth = 1, 
                    linecolor = 'black',
                    tickvals = np.arange(0, 1.1, 0.1),
                    range = [-0.01,1.02],
                    title = 'relative depth',
                    ticks = 'outside',
                    row = row,
                    col = col,
                    ) 

    fig.update_yaxes(showline = True, 
                    linewidth = 1, 
                    linecolor = 'black', 
                    tickvals = np.arange(0, 1.1, 0.1), 
                    range = [0,1.02], 
                    title = title, 
                    ticks = 'outside',
                    row = row,
                    col = col,
                    )
    return fig


