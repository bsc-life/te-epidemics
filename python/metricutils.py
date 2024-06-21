import xarray as xr
import networkx as nx
import numpy as np

def get_top_related(reference, Txy, k=5, agg_by="sum", relation="source"):
    if relation == "source":
        xa = Txy.loc[:, reference, :]
    elif relation == "target":
        xa = Txy.loc[reference, :, :]   
    
    if agg_by == 'sum':
        top_k = xa.sum("date").to_pandas()
    elif agg_by == 'max':
        top_k = xa.max("date").to_pandas()
        
    top_k = top_k.loc[top_k > 0]
    top_k = top_k.sort_values(ascending=False)
    
    return top_k[:k].index
    
def calculate_TSxy(Txy):
    TSxy = Txy.values - Txy.values.transpose(1, 0, 2)
    TSxy = xr.DataArray(TSxy, dims=Txy.dims, coords=Txy.coords)
    return TSxy

def calculate_net_TSxy(Txy):
    TSxy = calculate_TSxy(Txy)
    TSxy = xr.where(TSxy > 0, TSxy, 0)
    return TSxy
    
def calculate_normalized_TSx(Txy):
    TSxy = calculate_TSxy(Txy)
    TSx = TSxy.sum("target")

    pos_TSx = xr.where(TSx > 0, TSx, 0)
    pos_TSx = pos_TSx / pos_TSx.max("source")
    pos_TSx = xr.where(np.isnan(pos_TSx), 0, pos_TSx)

    neg_TSx = xr.where(TSx < 0, TSx, 0)
    neg_TSx = -1 * (neg_TSx / neg_TSx.min("source"))
    
    neg_TSx = xr.where(np.isnan(neg_TSx), 0, neg_TSx)

    x_di = pos_TSx + neg_TSx
    return x_di

def calculate_normalized_net_TSxy(Txy):    
    Txy_total = Txy.values + Txy.values.transpose(1, 0, 2)
    Txy_total = xr.DataArray(Txy_total, dims=Txy.dims, coords=Txy.coords)
    Txy_total /= 2

    net_TSxy = calculate_net_TSxy(Txy)
    normalized_net_TSxy = xr.where(net_TSxy > 0, net_TSxy/Txy_total, 0)
    return normalized_net_TSxy
    
def create_TSxy_network(Txy, data, date, thr_filter=0.5):
    TSxy = calculate_TSxy(Txy)
    A = TSxy.loc[:,:,date].to_pandas()
    A = A.T.unstack()
    A = A[A > A.max() * thr_filter]
    G = nx.DiGraph()
    G.add_nodes_from(data.index)
    G.add_weighted_edges_from(A.reset_index().values)
    return G
