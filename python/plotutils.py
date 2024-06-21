import numpy as np
import pandas as pd
import networkx as nx
import xarray as xr
import string
import datetime
import sys

from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.affinity import affine_transform

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("../python/")

from metricutils import *

################################################################################################################################
##########################################                PROVINCES                #############################################
################################################################################################################################

def plot_param_sweep(df, omega_best):
    fig,ax = plt.subplots(1, 1, figsize=(8,4.5) ,dpi=150,sharex=True)
    fig.set_facecolor('white')
    ax.set_ylabel("Total net TE")
    
    left, bottom, width, height = [0.6, 0.7, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.set_ylabel("Total net TE")

    sns.color_palette("muted")
    data = df.copy()
    data["delta"] = data["delta"].astype(str)
    sns.scatterplot(data=data,x="omega",y="total_di",hue="delta",ax=ax,style="delta")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    df_omega = df.loc[df.omega == omega_best]
    sns.scatterplot(data=df_omega,x="delta",y="total_di",ax=ax2)
    
    fig.tight_layout()
	
    return fig
    
def plot_top_k(Txy, risk_ds, cases_ds, prov_names, relation, reference, k, max_te, max_risk, date_range):
    m = 10
    n = int( (k+1) * 1.5 )
    trimed_Txy = Txy.loc[:,:,f"2020-{date_range[0]}-30":f"2020-{date_range[1]}-30"]
    trimed_dates = trimed_Txy.coords['date'].values

    figsize = (m,n)
    fig, axes = plt.subplots((k+1), 1, figsize=figsize, dpi=100, sharex=True)
    fig.set_facecolor('white')

    agg_by = 'sum'
    top_k =  get_top_related(reference, trimed_Txy, k=k, agg_by=agg_by, relation=relation)

    for i,related in enumerate(top_k):  
        ax = axes[i]
        var_name = r'$TE_{xy}$'
    
        if relation == "source":
            ts = Txy.loc[related, reference, trimed_dates].to_pandas()
        elif relation == "target":
            ts = Txy.loc[reference, related, trimed_dates].to_pandas()
        
        ts = pd.DataFrame(ts, columns=[var_name])
    
        sns.lineplot(x="date", y=var_name, data=ts, ax=ax, color='red', label=r"$+DI_{xy}(t)$")
        ax.set_ylabel(r"$+DI_{xy}(t)$")
        ax.legend(loc=2)
        ax.set_ylim(0,max_te)
        
        ax = ax.twinx()
        var_name = "risk_ij"
        if relation == "source":
            ts = risk_ds[var_name].loc[related, reference, trimed_dates].to_pandas()
        elif relation == "target":
            ts = risk_ds[var_name].loc[reference, related, trimed_dates].to_pandas()
        ts = pd.DataFrame(ts, columns=[var_name])
        ts = ts.rolling(21, min_periods=1).sum()
        sns.lineplot(x="date", y=var_name, data=ts, ax=ax, color='blue', label=r"$R_{xy}(t)$"+f" {prov_names[related]}")
        ax.set_ylabel(r"${R}_{xy}(t)$")
        ax.legend(loc=1)
        ax.set_ylim(0,max_risk)

    ax = axes[-1]
    ts = cases_ds['new_cases_by_100k'].loc[reference, trimed_dates].to_pandas()
    ax.fill_between(ts.index, ts.values, color='grey', alpha=0.2)
    ax.plot(ts.index,ts.values, c='black')
    ax.set_ylabel(r'$I_y by 100k$')

    for i in range(len(axes)):
        ax = axes[i]
        ax.text(-0.025, 1.08, string.ascii_lowercase[i], fontsize=14, transform=ax.transAxes, weight='bold', color='#333333')

    ax.set_xlim([datetime.date(2020,int(date_range[0]),30), datetime.date(2020,int(date_range[1]),30)])
    axes[0].set_title(f"Top-{k} TE {relation}s for {prov_names[reference]}")
    axes[3].set_title(f"New cases by 100k inhabitants for {prov_names[reference]}")
    fig.tight_layout()

    return fig
    
def plot_heatmap(DIx, prov_names, sort_values=True, ax=None, linewidths=0.01, cmap=mpl.cm.bwr, top_x=0):
    df_DIx = DIx.to_pandas()
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(16,7), dpi=300)
        
    if sort_values:
        net_DI = xr.where(DIx > 0, DIx, 0)
        net_DI = net_DI.to_pandas()
        idx = net_DI.sum(1).sort_values(ascending=False).index
        if top_x > 0:
            df_DIx = df_DIx.loc[idx[:top_x]]
        else:
            df_DIx = df_DIx.loc[idx]

    sns.heatmap(df_DIx, cmap=cmap, linewidths=linewidths, linecolor='white', ax=ax,
                yticklabels=[prov_names[x] for x in df_DIx.index], cbar_kws={"shrink": .82, "pad": 0.01}, cbar=False, vmin=-1, vmax=1)

    ax.tick_params(axis='y', which='major', labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    return ax

def plot_heatmap_cases(Txy, df_cases, prov_names, sort_values=True, scale_values=False, figsize=(32,18), linewidths=0.01, cmap=mpl.cm.bwr, top_x=0):
    if scale_values:
    	DIx = calculate_normalized_TSx(Txy)
    else:
        TSxy = calculate_TSxy(Txy)
        DIx = TSxy.sum("target")
    df_DIx = DIx.to_pandas()

    n = Txy.coords["date"].shape[0]
    m = Txy.coords["source"].shape[0]
    if top_x > 0:
        m = top_x

    if m < 100:
        figsize = (16, 10)
        height_ratios = [6, 1.35]
    elif m < 500:
        figsize = (32, 70)
        height_ratios = [12, 1]
    
    fig, axes = plt.subplots(2,1, figsize=figsize, dpi=300, gridspec_kw={'height_ratios': height_ratios} )
    fig.set_facecolor('white')
    
    plot_heatmap(DIx, prov_names, sort_values=sort_values, ax=axes[0], linewidths=linewidths, cmap=cmap, top_x=top_x)
    axes[0].set_xticklabels([])
    
    df = df_cases.reset_index()
    for prov in prov_names.keys():
        sns.lineplot(data=df.loc[df.id==prov], x="date", y="new_cases_by_100k", ax=axes[1], color='black', linewidth=0.6, alpha=0.5)
    axes[1].set_xlim((df_DIx.columns[0], df_DIx.columns[-1]))
    axes[1].set_ylabel("New cases by 100K individuals", fontsize=8)
    axes[1].tick_params(axis='both', which='major', labelsize=8)
    axes[1].tick_params(axis='x', which='major', labelsize=10)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlabel("")

    #netTSxy = calculate_net_TSxy(Txy)
    #netTSxy.sum("target").sum("source").plot(label="total net TE", ax=axes[2])

    axes[0].text(-0.08, 0.99, "A", fontsize=18, 
                        transform=axes[0].transAxes, weight='bold', color='#333333')
    axes[1].text(-0.08, 1.0, "B", fontsize=18,
                        transform=axes[1].transAxes, weight='bold', color='#333333')

    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    fig.tight_layout()
    
    return fig

def plot_network_on_map(G, gdf_patches, cases, date=""):

    norm = plt.Normalize(0, cases.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "red"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor('white')

    ax.set_axis_off()
    ax.set_facecolor('white')
    ax.set_title(f"Mobility Associated Risk {date}")

    gdf_patches.plot(edgecolor='black', linewidth=0.1, ax=ax, color=cases.map(norm).map(cmap))

    coord_dict = {}
    for i in gdf_patches.index:
        pro = gdf_patches.loc[i, 'geometry']
        if type(pro) is MultiPolygon:
            idx = np.argmax([p.area for p in pro.geoms])
            pro = pro.geoms[idx]
        x = pro.centroid.x
        y = pro.centroid.y
        coord_dict[i] = (x, y)

    nodes_params = dict(pos=coord_dict, node_size=5, node_color=cases.map(norm).map(cmap))
    nodes_params = dict(pos=coord_dict, node_size=5, node_color="grey")

    nodes = nx.draw_networkx_nodes(G, **nodes_params)

    edges = nx.draw_networkx_edges(G, coord_dict, arrowstyle='-|>',
                                connectionstyle="arc3,rad=0.4", arrowsize=6,
                                width=0.4, node_size=12)

    return fig


    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.01,
                        location='bottom')

    cbar.set_label("Daily cases by 100K")


    DIxy = calculate_directionality_index(Txy)

def plot_DI_map(DIxy, gdf_patches, move_islands=False):

    DIx = DIxy.sum('target')
    DIx = DIx.sum('date')

    DIx = DIx.to_pandas()

    data = DIx.values
    data = normalize_DI(data)
    DIx = pd.Series(data.reshape(1, -1)[0], index=DIx.index)
    DIx.name = 'total_dix'

    gdf = pd.merge(gdf_patches, DIx, left_index=True, right_index=True)
    gdf = gdf[['total_dix', 'geometry']].copy()

    if move_islands:
        xoff = 5
        yoff = 6.5
        matrix = [1, 0, 0, 1, xoff, yoff]
        for i in ["38", "35"]:
            geom = gdf.loc[i, 'geometry']
            gdf.at[i, 'geometry'] = affine_transform(geom, matrix)

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    #cmap = sns.color_palette("vlag", as_cmap=True, norm=norm)
    cmap = mpl.cm.bwr

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('white')

    ax.set_axis_off()
    ax.set_facecolor('white')
    gdf.plot(edgecolor='black', linewidth=0.3, column='total_dix', cmap=cmap, ax=ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, fraction=0.05, pad=0.5, orientation="horizontal")#, boundaries=[-1.,0.,1.]), values=[-1.,1.])

    cbar.set_label("Total DIx")
    
    fig.tight_layout()

    return fig
    
def plot_map_net(ax, gdf, G, column, data, date, layer, vmax=None):
    
    if layer == "cnig_provincias":
        xoff = 5;  yoff = 7.2
        matrix = [1, 0, 0, 1, xoff, yoff]
        canary_islands = ["38", "35"]
        for i in canary_islands:
            geom = gdf.loc[i, 'geometry']
            gdf.at[i, 'geometry'] = affine_transform(geom, matrix)
            centroid = gdf.loc[i, 'centroid']
            gdf.at[i, 'centroid'] = affine_transform(centroid, matrix)
        ax.add_patch(Rectangle((-13.5,34.6), 5.5, 2.2, edgecolor='silver', facecolor='none', lw=1))

    cmap = sns.color_palette("Purples", as_cmap=True)
    if vmax is not None:
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=0, vmax=data.max())
    
    gdf.plot(edgecolor='silver', linewidth=0.5, column=column, cmap=cmap, ax=ax, norm=norm)

    ax.set_axis_off()
    ax.set_facecolor('white')
    
    coord_dict = {k:(v.x, v.y) for k,v in gdf['centroid'].to_dict().items()}
    nodes = nx.draw_networkx_nodes(G, coord_dict, node_size=0, node_color="white", ax=ax)
    edges = nx.draw_networkx_edges(G, coord_dict, arrowstyle='fancy', connectionstyle="arc3,rad=0.24", 
                               arrowsize=5, node_size=40, edge_color="firebrick", alpha=0.5, ax=ax)
    nodes.set_edgecolor('grey')

################################################################################################################################
##########################################                TOY MODEL                #############################################
################################################################################################################################

def plot_pop_on_graph(mov, patches, nodes_dict, topology, ax=None, hub_idx="M1"):
    edges_list = []
    for i,pi in enumerate(patches):
        for j,pj in enumerate(patches):
            if mov[i,j] != 0 and i != j:
                edges_list.append((nodes_dict[pi],nodes_dict[pj], mov[i,j]))
                
    G = nx.DiGraph()
    G.add_nodes_from([nodes_dict[x] for x in patches])

    for item in edges_list:
        G.add_edge(item[0],item[1],weight=int(item[2]))

    if topology == "star":
        center_node="M1"
        edge_nodes = set(G) - {center_node}
        # Ensures the nodes around the circle are evenly distributed
        pos = nx.circular_layout(G.subgraph(edge_nodes))
        pos[center_node] = np.array([0, 0])  # manually specify node position
    elif topology == "ring":
        pos = nx.circular_layout(G)
    elif topology == "chain":
        pos = nx.shell_layout(G)
        
    node_sizes = 750

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="slategrey", ax=ax)
    hub = nx.draw_networkx_nodes(G, pos, nodelist=[hub_idx], node_size=node_sizes, node_color="firebrick", ax=ax)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        #add_labels=True,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        connectionstyle="arc3, rad=0.2",
        width=1.5,
        ax=ax
    )
    labels={nodes_dict[y]:nodes_dict[y] for (x,y) in enumerate(patches)}
    edge_labels = nx.get_edge_attributes(G,'weight')
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="lightgrey", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, bbox=dict(alpha=0), ax=ax)
    
    ax.set_axis_off()

    return
    
def plot_TE_on_graph(TE_ds, patches, nodes_dict, topology, ax=None, hub_idx="M1", date=None, colorbar=True):
    if date == None:
        te_date = TE_ds.sum(dim="date")
    else:
        te_date = TE_ds.loc[:,:,date]

    max_te = TE_ds.max().values.item()

    edges_list = []
    for i in patches:
        for j in patches:
            if i != j:
                if date != None:
                    edges_list.append((nodes_dict[i],nodes_dict[j], te_date.loc[i,j].values.item()/max_te))
                else:
                    edges_list.append((nodes_dict[i],nodes_dict[j], te_date.loc[i,j].values.item()/11))

    G = nx.DiGraph()
    G.add_nodes_from([nodes_dict[x] for x in patches])

    for item in edges_list:
        if item[2] > 0:
            G.add_edge(item[0],item[1],weight=item[2])

    if topology == "star":
        center_node="M1"
        edge_nodes = set(G) - {center_node}
        # Ensures the nodes around the circle are evenly distributed
        pos = nx.circular_layout(G.subgraph(edge_nodes))
        pos[center_node] = np.array([0, 0])  # manually specify node position
    elif topology == "ring":
        pos = nx.circular_layout(G)
    elif topology == "chain":
        pos = nx.circular_layout(G)
    node_sizes = 750
    cmap = plt.cm.Reds

    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    edge_colors = [cmap(weight) for weight in weights]

    #fig = plt.figure(figsize=(6,5),dpi=300)

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="slategrey", ax=ax)
    hub = nx.draw_networkx_nodes(G, pos, nodelist=[hub_idx], node_size=node_sizes, node_color="firebrick", ax=ax)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
        edge_color=edge_colors,
        #edge_cmap=cmap,
        width=2,
        ax=ax
    )
    labels={nodes_dict[y]:nodes_dict[y] for (x,y) in enumerate(patches)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="lightgrey", ax=ax)

    if colorbar:
        pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        pc.set_array(edge_colors)
        pc.set_clim(0,1)
        plt.colorbar(pc, ax=ax, label=r"norm. $DI_{XY}$")

    plt.sca(ax)
    #ax = plt.gca()
    #ax.set_axis_off()

    return
    
def plot_TE_on_graph2(DIxy, patches, nodes_dict, topology, pos, ax=None, hub_idx="M1", date=None, 
                      colorbar=True, cbar_type="nodes", nodes_color=None):
    if date == None:
        DI_date = DIxy.sum(dim="date")
    else:
        DI_date = DIxy.loc[:,:,date]

    max_te = DIxy.max().values.item()
    
    edges_list = []
    for i in patches:
        for j in patches:
            if i == j:
                continue
            
            if date is not None:
                edges_list.append((nodes_dict[i],nodes_dict[j], DI_date.loc[i,j].values.item()/max_te))
            else:
                edges_list.append((nodes_dict[i],nodes_dict[j], DI_date.loc[i,j].values.item()/11))

    G = nx.DiGraph()
    G.add_nodes_from([nodes_dict[x] for x in patches])

    for item in edges_list:
        if item[2] > 0:
            G.add_edge(item[0],item[1],weight=item[2])
    
    node_sizes = 500
    cmap = plt.cm.Reds

    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    edge_colors = [cmap(weight) for weight in weights]


    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=nodes_color, ax=ax, edgecolors='black')
    
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
        edge_color=edge_colors,
        #edge_cmap=cmap,
        width=2,
        ax=ax
    )
    labels={nodes_dict[y]:nodes_dict[y] for (x,y) in enumerate(patches)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="lightgrey", ax=ax)

    if colorbar:
        if cbar_type == "edges":
            pc = mpl.collections.PatchCollection(edges, cmap=cmap)
            pc.set_array(edge_colors)
            pc.set_clim(0,1)
            plt.colorbar(pc, ax=ax, label=r"norm. $DI_{XY}$")
        else:
            pc = mpl.collections.PathCollection(nodes, cmap=sns.color_palette("vlag", as_cmap=True))#cmap=plt.cm.Purples
            pc.set_array(nodes_color)
            pc.set_clim(-1,1)
            plt.colorbar(pc, ax=ax, label=r"$DI_{X}$", ticks=[-1,-0.5,0,0.5,1])
            #pc.set_clim(0, cases_ds["new_cases_by_100k"].max())
            #plt.colorbar(pc, ax=ax, label=r"Active % of infected individuals")

    plt.sca(ax)
    ax = plt.gca()
    ax.set_axis_off()

    return
    
def plot_incidence(cases_ds, days_interval, n_days, ax, nodes_dict):
    dates_10 = ["2020-03-07","2020-03-17","2020-03-27",
                "2020-04-06", "2020-04-16", "2020-04-26",
                "2020-05-06", "2020-05-16", "2020-05-26",
                "2020-06-05", "2020-06-15", "2020-06-25"]
    
    dates_30 = ["2020-03-07", "2020-04-06", "2020-05-06",
                "2020-06-05", "2020-07-05", "2020-08-04",
                "2020-09-03", "2020-10-03"]
    
    n = int(n_days/days_interval)+1
    if days_interval == 10:
        dates = dates_10[:n]
    elif days_interval == 30:
        dates = dates_30[:n]
        
    days = ["D"+str(x) for x in range(0,n_days+1,days_interval)]
    
    cp = sns.color_palette("deep", 9)
    
    for idx,i in enumerate(cases_ds.coords['id']):
        ax.plot(cases_ds["new_cases_by_100k"].loc[i,dates[0]:dates[-1]].to_pandas(), label=nodes_dict[str(i.values)],
                  color=cp[idx])
        
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("New cases by 10k people")
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.set_xticks(dates)
    ax.set_xticklabels(days, fontsize=10)
    
    return (dates[0], dates[-1])
    
def plot_toymodel_heatmap(Txy, ax):
    data = calculate_normalized_TSx(Txy)[:,:n_days]
    DIx = data.to_pandas()

    ticks = [nodes_dict[x] for x in DIx.index]
    sns.heatmap(DIx, cmap=sns.color_palette("RdBu_r",as_cmap=True), linewidths=0.01, linecolor='white', ax=ax,
            yticklabels=ticks, xticklabels="", cbar_kws={"label": "$DI_x$", "ticks":[-1,-0.5,0,0.5,1]},
            cbar=True, vmin=-1, vmax=1)

    ax.set_yticklabels(ticks, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    return
