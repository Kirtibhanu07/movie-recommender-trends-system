"""Sparse graph recommendation and leakage-aware offline ranking utilities."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mplconfig"))


def load_dataset(folder=ROOT/"data"):
    ratings=pd.read_csv(folder/"ratings.dat",sep="::",engine="python",names=["user_id","movie_id","rating","timestamp"])
    movies=pd.read_csv(folder/"movies.dat",sep="::",engine="python",names=["movie_id","title","genres"],encoding="latin1")
    if ratings.isna().any().any() or movies.isna().any().any(): raise ValueError("Missing required data")
    if ratings.duplicated(["user_id","movie_id"]).any(): raise ValueError("Duplicate user/movie rating")
    if movies.movie_id.duplicated().any(): raise ValueError("Duplicate movie ID")
    if not ratings.rating.between(1,5).all(): raise ValueError("Ratings must be 1-5")
    if not set(ratings.movie_id).issubset(set(movies.movie_id)): raise ValueError("Unknown rated movie")
    users=np.sort(ratings.user_id.unique())
    movies=movies.sort_values("movie_id").reset_index(drop=True)
    umap={int(v):i for i,v in enumerate(users)};imap={int(v):i for i,v in enumerate(movies.movie_id)}
    ratings['u']=ratings.user_id.map(umap);ratings['i']=ratings.movie_id.map(imap)
    movies['year']=pd.to_numeric(movies.title.str.extract(r"\((\d{4})\)$")[0],errors="coerce").fillna(0).astype(int)
    return ratings,movies,users


def chronological_split(ratings):
    """Never split a timestamp group. Do not invent order within a rating session."""
    pieces=[]
    for _,g in ratings.groupby('u',sort=True):
        g=g.sort_values(['timestamp','i']).copy();times=np.sort(g.timestamp.unique());n=len(times)
        if n<3: g['split']='train'
        else:
            a=max(1,min(int(.70*n),n-2));b=max(a+1,min(a+max(1,int(.15*n)),n-1))
            g['split']=np.where(g.timestamp<times[a],'train',np.where(g.timestamp<times[b],'validation','test'))
        pieces.append(g)
    return pd.concat(pieces,ignore_index=True)


def positive_matrix(frame,n_users,n_items):
    pos=frame[frame.rating>=4]
    matrix=sparse.csr_matrix((np.ones(len(pos),np.float32),(pos.u,pos.i)),shape=(n_users,n_items))
    matrix.sum_duplicates();matrix.data[:]=1
    return matrix


def seen_matrix(frame,n_users,n_items):
    matrix=sparse.csr_matrix((np.ones(len(frame),np.float32),(frame.u,frame.i)),shape=(n_users,n_items))
    matrix.sum_duplicates();matrix.data[:]=1
    return matrix


def genre_matrix(movies):
    genres=sorted({g for row in movies.genres for g in row.split('|')})
    indices={g:i for i,g in enumerate(genres)}
    features=np.zeros((len(movies),len(genres)),np.float32)
    for i,row in enumerate(movies.genres):
        for g in row.split('|'): features[i,indices[g]]=1
    idf=np.log((1+len(movies))/(1+features.sum(axis=0)))+1
    return normalize(features*idf).astype(np.float32),genres


def normalized_graph(positives):
    n_users,n_items=positives.shape
    coo=positives.tocoo();left=coo.row;right=coo.col+n_users
    rows=np.concatenate([left,right]);cols=np.concatenate([right,left])
    degree=np.bincount(rows,minlength=n_users+n_items)
    inv=np.zeros_like(degree,dtype=np.float32);mask=degree>0;inv[mask]=1/np.sqrt(degree[mask])
    values=inv[rows]*inv[cols]
    return torch.sparse_coo_tensor(torch.tensor(np.stack([rows,cols]),dtype=torch.long),
                                  torch.tensor(values,dtype=torch.float32),size=(n_users+n_items,)*2).coalesce()


class GraphRecommender(nn.Module):
    def __init__(self,n_users,n_items,dimension=32,layers=2):
        super().__init__();self.n_users=n_users;self.n_items=n_items;self.layers=layers
        self.embedding=nn.Embedding(n_users+n_items,dimension)
        nn.init.normal_(self.embedding.weight,std=.1)

    def propagate(self,graph):
        current=self.embedding.weight;representations=[current]
        for _ in range(self.layers):
            current=torch.sparse.mm(graph,current);representations.append(current)
        return torch.stack(representations).mean(dim=0)

    def bpr_loss(self,graph,users,positive,negative,l2=.0001):
        all_vectors=self.propagate(graph)
        u=all_vectors[users];p=all_vectors[positive+self.n_users];n=all_vectors[negative+self.n_users]
        ranking=-F.logsigmoid((u*p).sum(1)-(u*n).sum(1)).mean()
        raw=self.embedding(torch.cat([users,positive+self.n_users,negative+self.n_users]))
        return ranking+l2*raw.square().sum()/len(users)


def sample_triplets(positives,batch_size,rng):
    if np.any(np.asarray(positives.sum(1)).ravel() >= positives.shape[1]):
        raise ValueError('Cannot sample negatives for a user who liked the entire catalogue')
    coo=positives.tocoo();idx=rng.integers(0,len(coo.data),batch_size)
    u,p=coo.row[idx],coo.col[idx];n=rng.integers(0,positives.shape[1],batch_size)
    # Only training positives are excluded. Do not consult future labels.
    invalid=np.asarray(positives[u,n]).ravel()>0
    while invalid.any():
        n[invalid]=rng.integers(0,positives.shape[1],invalid.sum())
        invalid=np.asarray(positives[u,n]).ravel()>0
    return tuple(torch.tensor(x,dtype=torch.long) for x in [u,p,n])


def ranked_topk(scores,k=20):
    if not 1<=k<=len(scores): raise ValueError("Invalid top-k")
    # Stable movie-index tie breaking. All masked scores remain negative infinity.
    return np.argsort(-scores,kind='stable')[:k]


def ranking_metrics(ranking,relevant):
    relevant=set(relevant)
    if not relevant: raise ValueError("No positive target")
    out={}
    for k in [10,20]:
        hit=np.array([i in relevant for i in ranking[:k]],float)
        ideal=np.sum(1/np.log2(np.arange(2,min(k,len(relevant))+2)))
        out[f'NDCG@{k}']=float(np.sum(hit/np.log2(np.arange(2,len(hit)+2)))/ideal)
        out[f'Recall@{k}']=float(hit.sum()/len(relevant))
        if k==20:
            out['Precision@20']=float(hit.sum()/20);out['HitRate@20']=float(hit.sum()>0)
    return out


def score_components(user_vectors,item_vectors,profiles,genres,users):
    return dot_products(user_vectors[users],item_vectors), dot_products(profiles[users],genres)


def dot_products(left,right):
    """Use one checked float32 scoring backend for training evaluation and serving."""
    with torch.inference_mode():
        scores=(torch.as_tensor(np.asarray(left,dtype=np.float32)) @
                torch.as_tensor(np.asarray(right,dtype=np.float32)).T).numpy()
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite recommendation scores')
    return scores


def hybrid_scores(collaborative,content,weight):
    if weight==0: return collaborative.copy()
    scale=np.maximum(collaborative.std(axis=1,keepdims=True),1e-6)
    normalized_cf=(collaborative-collaborative.mean(axis=1,keepdims=True))/scale
    return normalized_cf+weight*content


def evaluate_scores(name,users,score_fn,seen,truth,k=20):
    rows=[];covered=set()
    for start in range(0,len(users),256):
        batch=users[start:start+256];scores=np.asarray(score_fn(batch),dtype=np.float32).copy()
        if not np.isfinite(scores).all(): raise ValueError('Nonfinite scores before masking')
        for j,u in enumerate(batch):
            scores[j,seen[u].indices]=-np.inf
            ranking=ranked_topk(scores[j],k)
            relevant=set(truth[u].indices)-set(seen[u].indices)
            if not relevant: continue
            covered.update(ranking.tolist())
            rows.append({'model':name,'user_index':int(u),**ranking_metrics(ranking,relevant)})
    frame=pd.DataFrame(rows)
    if frame.empty: raise ValueError("No eligible evaluation users")
    aggregate={"model":name,"users":len(frame),**frame.drop(columns=['model','user_index']).mean().to_dict(),
               'catalogue_coverage@20':len(covered)/seen.shape[1]}
    return aggregate,frame


def paired_bootstrap(selected,popularity,n=10000):
    a=selected.set_index('user_index')['NDCG@20'];b=popularity.set_index('user_index')['NDCG@20'].reindex(a.index)
    if b.isna().any(): raise ValueError("Unpaired users")
    delta=(a-b).to_numpy();rng=np.random.default_rng(42)
    values=[]
    for _ in range(n//100):
        idx=rng.integers(0,len(delta),size=(100,len(delta)))
        values.extend(delta[idx].mean(axis=1).tolist())
    return {'users':len(delta),'mean_NDCG20_difference':float(delta.mean()),
            '95pct_paired_user_bootstrap_interval':np.quantile(values,[.025,.975]).tolist(),
            'resamples':n,'seed':42,'limit':'One training seed and one retrospective split. Not a production A/B test.'}


def content_profile(positives,genres):
    return normalize(np.asarray(positives@genres)).astype(np.float32)
