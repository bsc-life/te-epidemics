import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import networkx as nx

def normalize_DI(X):

    pos_X = X*(X >= 0)
    neg_X = -X*(X < 0)

    pos_X_std = (pos_X - pos_X.min(axis=0)) / (pos_X.max(axis=0) - pos_X.min(axis=0))
    neg_X_std = (neg_X - neg_X.min(axis=0)) / (neg_X.max(axis=0) - neg_X.min(axis=0))

    return pos_X_std - neg_X_std

def plot_heatmap(df_DIx, sort_values=True, scale_values=False, linewidths=0.01, cmap=mpl.cm.bwr):

    fig, ax = plt.subplots(1,1, figsize=(32,14), dpi=300)

    if scale_values:
        data = normalize_DI(df_DIx)
        df_DIx = pd.DataFrame(data=data, columns=df_DIx.columns, index=df_DIx.index)

    if sort_values:
        idx = df_DIx.sum(1).sort_values(ascending=False).index
        df_DIx = df_DIx.loc[idx]

    sns.heatmap(df_DIx, cmap=cmap, linewidths=linewidths, linecolor='white', ax=ax,
                yticklabels=df_DIx.index, cbar_kws={"shrink": .82, "pad": 0.01}, cbar=False, vmin=-1, vmax=1)

    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")

    return fig

def plot_TE_on_graph(TE_ds, patches, subpopulation_sizes, topology):
    te_date = TE_ds[:,:,1]
    te_date = TE_ds.sum(dim="date")
    edges_list = []
    for i in patches:
        for j in patches:
            if i != j:
                edges_list.append((i,j, te_date.loc[i,j].values.item()))
		
    G = nx.DiGraph()
    G.add_nodes_from(patches)

    for item in edges_list:
        if item[2] > 0:
            G.add_edge(item[0],item[1],weight=item[2])

    if topology == "star":
        center_node = "H"
        edge_nodes = set(G) - {center_node}
        # Ensures the nodes around the circle are evenly distributed
        pos = nx.circular_layout(G.subgraph(edge_nodes))
        pos[center_node] = np.array([0, 0])  # manually specify node position
    elif topology == "ring":
        pos = nx.circular_layout(G)
    elif topology == "chain":
        pos = nx.circular_layout(G)
    node_sizes = subpopulation_sizes/10
    cmap = plt.cm.Reds

    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    edge_colors = weights

    fig = plt.figure(figsize=(8,6),dpi=300)

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )
    labels={y:y for (x,y) in enumerate(patches)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="lightgrey")

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax, label="Transfer Entropy")

    return ax

def plot_incidence(da_inf):
    df = da_inf.mean(dim="rep").to_pandas()

    fig,ax = plt.subplots(figsize=(6,3), dpi=200)
    for c in df.columns:
        ax.plot(df.index, df[c],label=c)
    ax.legend()
    plt.ylabel("Cases")
    sns.despine(fig)

    return fig
