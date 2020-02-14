import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
from pyproj import Proj, transform
import mplleaflet
import numpy as np

import torch

def make_enc_map(model_type, cluster_labels, num_cluster, extent, margin,
                 coords_mat = None, coords_color = "red", colorbar=False, img_path=None, xlabel = None, ylabel = None):
    cmap = plt.cm.terrain
    bounds = np.arange(-0.5,num_cluster + 0.5,1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(32, 32))

#     pt_x_list, pt_y_list = plot_poi_by_type(enc_dec, type2pts, tid)
    plt.matshow(cluster_labels[::-1, :], extent=extent, cmap=cmap, norm = norm);
    # plt.colorbar()

    # We must be sure to specify the ticks matching our target names
    if colorbar:
        plt.colorbar(ticks=bounds-0.5)
    if coords_mat is not None:
        if coords_mat.shape:
            plt.scatter(coords_mat[:, 0], coords_mat[:, 1], s=1.5, c=coords_color, alpha=0.5)

#     plt.scatter(pt_x_list, pt_y_list, s=1.5, c="red", alpha=0.5)
    plt.xlim(extent[0]-margin, extent[1]+margin)
    plt.ylim(extent[2]-margin, extent[3]+margin)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
     
    if model_type == "global":
        plt.xticks(np.arange(extent[0]-margin, extent[1]+margin, 10000))
    fig = plt.gcf()
    # fig.suptitle(tid2type[tid])
#     plt.xlabel(poi_type, fontsize=10)
    plt.show()
    if img_path:
        fig.savefig(img_path, dpi=300)

def spa_enc_embed_clustering(enc_dec, dataset, model_type, spa_enc, num_cluster, interval, coords, tsne_comp = 4):

    
    # shape: (num_y*num_x, embed_dim)
    res = enc_dec.spa_enc.forward(coords)

    res_data = res.data.tolist()
    res_np = np.asarray(res_data)

    num_y,num_x,embed_dim = res_np.shape
    embeds = np.reshape(res_np, (num_y*num_x, -1))
    
#     embeds = TSNE(n_components=tsne_comp).fit_transform(embeds)
    
    
    embed_clusters = AgglomerativeClustering(n_clusters=num_cluster, affinity="cosine", linkage="complete").fit(embeds)
    cluster_labels = np.reshape(embed_clusters.labels_, (num_y, num_x))

    return embeds, cluster_labels

def spa_enc_head_embed_clustering(enc_dec, dataset, model_type, spa_enc, num_cluster, interval, coords, tsne_comp = 4):

    
    # shape: (num_y*num_x, embed_dim)
#     res = enc_dec.spa_enc.forward(coords)
    res = enc_dec.spa_enc.make_input_embeds(coords)


    res_np = res

    num_y,num_x,embed_dim = res_np.shape
    embeds = np.reshape(res_np, (num_y*num_x, -1))
    
#     embeds = TSNE(n_components=tsne_comp).fit_transform(embeds)
    
    
    embed_clusters = AgglomerativeClustering(n_clusters=num_cluster, affinity="cosine", linkage="complete").fit(embeds)
    cluster_labels = np.reshape(embed_clusters.labels_, (num_y, num_x))

    return embeds, cluster_labels

def visualize_encoder(module, layername, coords, extent, num_ch = 8, img_path=None):
    if layername == "input_emb":
        res = module.make_input_embeds(coords)
        if type(res) == torch.Tensor:
            res = res.data.numpy()
        elif type(res) == np.ndarray:
            res = res
        print res.shape
        res_np = res
    elif layername == "output_emb":       
        res = module.forward(coords)
        embed_dim = res.size()[2]
        res_data = res.data.tolist()
        res_np = np.asarray(res_data)

    num_rows = num_ch/8
 
    plt.figure(figsize=(28, 50))

    for i in range(num_ch):
        if num_ch <= 8:
            ax= plt.subplot(1,num_ch ,i+1)
        else:
            ax= plt.subplot(num_rows,8 ,i+1)
        ax.imshow(res_np[:,:,i][::-1, :], extent=extent)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
#         plt.tight_layout()
#         plt.title(i, fontsize=160)
        plt.title(i, fontsize=40)
    fig = plt.gcf()
    plt.show()
    plt.draw()
    if img_path:
        fig.savefig(img_path, dpi=300, bbox_inches='tight')



def visualize_nn_all(enc_dec, layername, dataset, model_type, spa_enc, coords, extent):
    if layername == "input_emb":
        res = module.make_input_embeds(coords)
        print res.shape
        res_np = res
    elif layername == "output_emb":       
        res = module.forward(coords)
        embed_dim = res.size()[2]
        res_data = res.data.tolist()
        res_np = np.asarray(res_data)

    plt.figure(figsize=(128, 155))
    # plt.figure(figsize=(32, 155))
    # for i in range(embed_dim):
    for i in range(64):
        ax= plt.subplot(8,8 ,i+1)
        # ax= plt.subplot(1,8 ,i+1)
        ax.imshow(res_np[:,:,i][::-1, :], extent=extent)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # plt.tight_layout()
        plt.title(i, fontsize=160)
        
    plt.show()



def nn_dist_curve(enc_dec, dataset, model_type, spa_enc):
    radius = 10000
    extent = (-radius, radius, -radius, radius)
    coords = []
    interval  = 300
    for y in range(extent[0], extent[1]+interval, interval):
        coord = []
        for x in range(extent[2], extent[3]+interval, interval):
            coord.append([x,y])
        coords.append(coord)
    
    coords = np.asarray(coords)
    
    module = enc_dec.spa_enc
    res = module.forward(coords)
    # embed_dim = res.size()[2]
    res_data = res.data.tolist()
    res_np = np.asarray(res_data)
    
    num_y,num_x,embed_dim = res_np.shape
    # dist_mat: (num_y,num_x)
    dist_mat = np.sqrt(np.sum(np.power(coords, 2), axis = 2, keepdims = False))
    dist = dist_mat.reshape((num_y*num_x))
    

    act = res_np[:,:,10].reshape((num_y*num_x))
    plt.scatter(dist, act, s=0.5)

    
def g_spa_enc_embed_clustering(enc_dec, dataset, model_type, spa_enc, num_cluster, interval):
    coords = []
#     interval = 500
    extent = (-1710000, -1690000, 1610000, 1640000)
    # latitude
    for y in range(extent[2], extent[3]+interval, interval):
        coord = []
    #     longitude
        for x in range(extent[0], extent[1]+interval, interval):
            coord.append([x,y])
        coords.append(coord)
        
    
    # shape: (num_y*num_x, embed_dim)
    spa_embeds = enc_dec.g_spa_enc.forward(coords)
    num_y, num_x, _ = spa_embeds.size()
    spa_embeds = spa_embeds.view(num_y*num_x, -1)
    embeds = spa_embeds.data.tolist()
    
    
    embed_clusters = AgglomerativeClustering(n_clusters=num_cluster, affinity="cosine", linkage="average").fit(embeds)
    cluster_labels = np.reshape(embed_clusters.labels_, (num_y, num_x))

    return embeds, cluster_labels
