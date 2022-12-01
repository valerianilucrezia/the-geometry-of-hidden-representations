import numpy as np
import plotly.graph_objects as go


def create_ds(nlayer):
    ds = {i : {} for i in range(nlayer+1)}
    return ds


def add_data(ds, layer, distances, idxs, labels, knn):
    ds[layer]['distances'] = distances[:, :knn+1]
    ds[layer]['idxs'] = idxs[:, :knn+1]
    ds[layer]['labels'] = labels
    return ds


def overlap_mean(ds, k):
    overlap_alls = []

    for l in ds.keys():
        overlap = 0.

        distances = ds[l]['distances']
        nelem = distances.shape[0]
        idxs = ds[l]['idxs']
        gt_labels = ds[l]['labels']

        for i in range(nelem):
            neigh_idx_i = idxs[i, 1:k + 1]
            overlap += sum(gt_labels[neigh_idx_i] == gt_labels[i]) / k
        overlap = overlap / nelem
        overlap_alls.append(overlap)

    return overlap_alls


def overlap_layer():
    overlap_alls = []
    return overlap_alls


def plot_no(fig, layer, df, color, lab, legend=True):
    num = np.arange(layer+1)
    fig.add_trace(go.Scatter(x = num/(len(num)-1),
                            y = df,
                            mode = 'lines + markers',
                            name = lab,
                            marker = dict(color=color,size=8),
                            line = dict(color=color,width=1.5),
                            showlegend = legend
                        )
                )
    return fig
