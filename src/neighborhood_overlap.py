import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def get_data(path, label_path, layer, ng=100):
    ds = create_ds(layer)
    for i in np.arange(0, layer+1):
        neig = np.load(path + str(i) + '-neigh.npy')
        idx = np.load(path + str(i) + '-idx.npy')

        label = open(label_path + 'sp_lab.txt')
        label = np.array([l.strip('\n') for l in label])

        ds = add_data(ds, i, neig, idx, label, ng)
    return ds


def create_ds(nlayer):
    ds = {i : {} for i in range(nlayer+1)}
    return ds


def add_data(ds, layer, distances, idxs, labels, knn):
    ds[layer]['distances'] = distances[:, :knn+1]
    ds[layer]['idxs'] = idxs[:, :knn+1]
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

def update_figure(fig, title, save = False, name = ''):
    fig.update_xaxes(showline = True, 
                    linewidth = 1, 
                    linecolor = 'black',
                    tickvals = np.arange(0, 1.1, 0.1),
                    range = [-0.01,1.02],
                    title = 'relative depth',
                    ticks = 'outside'
                    )

    fig.update_yaxes(showline = True, 
                    linewidth = 1, 
                    linecolor = 'black', 
                    tickvals = np.arange(0, 1.1, 0.1), 
                    range = [0,1.02], 
                    title = title, 
                    ticks = 'outside'
                    )
    w = 600
    h = 500
    fig.update_layout(width = w, 
                    height = h, 
                    font = dict(size = 12),
                    legend = dict(orientation = "h",
                              yanchor = "top",
                              y = 1.12,
                              xanchor = "center",
                              x = 0.5, 
                              font = dict(size = 12)
                              )
                    ) 

    if save and name != '': 
        pio.write_image(fig, 
                        name, 
                        scale=5, 
                        width=w, 
                        height=h)
    return fig
