import numpy as np

mapping = {'esm1b':['ESM-1b','rgb(220,20,60)', np.arange(34)],
            'esm1v':['ESM-1v','rgb(0,139,139)',np.arange(34)],
            'ProtBert':['ProtBert','rgb(136, 78, 160)',np.arange(31)], 
            'ProtT5':['ProtT5-XL-U50','rgb(245, 148, 0)',np.arange(25)], 
            'esm2-15B':['ESM-2 (15B)','rgb(68, 1, 84)',np.arange(49)],
            'esm2-3B':['ESM-2 (3B)','rgb(65, 68, 135)',np.arange(37)],
            'esm2-650M':['ESM-2 (650M)','rgb(42, 120, 142)',np.arange(34)], 
            'esm2-150M':['ESM-2 (150M)','rgb(34, 168, 132)', np.arange(31)],
            'esm2-35M':['ESM-2 (35M)','rgb(122, 209, 81)',np.arange(13)], 
            'esm2-8M':['ESM-2 (8M)','rgb(253, 231, 37)',np.arange(7)], 
            'esm-MSA':['ESM-MSA-1b','rgb(66,145,31)',np.arange(13)]
            }

def update_figure(fig, w, h, save = False, name = ''):
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
