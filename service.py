"""Shared recommendation service for the UI and API. Safe array artifacts, no pickle loading."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
import torch
from recommender import ROOT,dot_products,hybrid_scores


class MovieService:
    def __init__(self,folder=ROOT/'artifacts'):
        torch.set_num_threads(4)
        self.folder=Path(folder)
        if not (self.folder/'serving.npz').exists(): raise FileNotFoundError('Train the model first with train_evaluate.py')
        with np.load(self.folder/'serving.npz',allow_pickle=False) as arrays:
            self.a={k:arrays[k] for k in arrays.files}
        self.movies=pd.read_csv(self.folder/'catalogue.csv')
        if not np.array_equal(self.movies.movie_id,self.a['movie_ids']): raise ValueError('Catalogue and model versions differ')
        self.seen=sparse.load_npz(self.folder/'seen.npz')
        self.likes=sparse.load_npz(self.folder/'training_likes.npz')
        self.config=json.loads((self.folder/'model_config.json').read_text())
        self.ids={int(v):i for i,v in enumerate(self.movies.movie_id)}
        self.users={int(v):i for i,v in enumerate(self.a['user_ids'])}
        self.item_unit=normalize(self.a['item_vectors']).astype(np.float32)

    def filter_mask(self,genres=None,year_min=None,year_max=None):
        mask=np.ones(len(self.movies),bool)
        if genres:
            requested=set(genres);available={g for s in self.movies.genres for g in s.split('|')}
            if requested-available: raise ValueError('Unknown genre')
            mask &= self.movies.genres.map(lambda s:bool(set(s.split('|'))&requested)).to_numpy()
        if year_min is not None: mask &= self.movies.year.to_numpy()>=year_min
        if year_max is not None: mask &= self.movies.year.to_numpy()<=year_max
        return mask

    def rank(self,scores,exclude,k,diversity=0.,genres=None,year_min=None,year_max=None,seeds=None):
        if not 1<=k<=50: raise ValueError('Choose 1 to 50 recommendations')
        if not 0<=diversity<=1: raise ValueError('Diversity must be between zero and one')
        if year_min is not None and year_max is not None and year_min>year_max: raise ValueError('Year range is reversed')
        valid=self.filter_mask(genres,year_min,year_max)
        valid[list(exclude)]=False
        candidates=np.flatnonzero(valid)
        if not len(candidates): return []
        ordered=candidates[np.argsort(-scores[candidates],kind='stable')][:max(100,k)]
        chosen=[]
        if diversity==0: chosen=ordered[:k].tolist()
        else:
            values=scores[ordered];scaled=(values-values.min())/max(float(np.ptp(values)),1e-8)
            remaining=list(range(len(ordered)))
            while remaining and len(chosen)<k:
                if not chosen: j=remaining[0]
                else:
                    similarity=dot_products(self.a['genre_vectors'][ordered[remaining]],self.a['genre_vectors'][chosen]).max(axis=1)
                    utility=(1-.65*diversity)*scaled[remaining]-.65*diversity*similarity
                    j=remaining[int(np.argmax(utility))]
                chosen.append(int(ordered[j]));remaining.remove(j)
        results=[]
        for rank,index in enumerate(chosen,1):
            row=self.movies.iloc[index];why='Based on learned preference patterns'
            if seeds:
                overlap=dot_products(self.a['genre_vectors'][[index]],self.a['genre_vectors'][seeds])[0]
                closest=seeds[int(np.argmax(overlap))]
                shared=sorted(set(row.genres.split('|'))&set(self.movies.iloc[closest].genres.split('|')))
                why=f"Shares {', '.join(shared[:3]) or 'preference signals'} with {self.movies.iloc[closest].title}"
            results.append({'rank':rank,'movie_id':int(row.movie_id),'title':row.title,'genres':row.genres.split('|'),
                            'year':int(row.year),'ranking_score':float(scores[index]),'reason':why})
        return results

    def favorites(self,movie_ids,k=10,diversity=.2,**filters):
        if not 1<=len(movie_ids)<=20: raise ValueError('Choose 1 to 20 favorites')
        if any(int(i) not in self.ids for i in movie_ids): raise ValueError('Unknown movie ID')
        seeds=sorted({self.ids[int(i)] for i in movie_ids});start=time.perf_counter()
        profile=self.item_unit[seeds].mean(axis=0,keepdims=True)
        cf=dot_products(profile,self.item_unit)
        genre=normalize(self.a['genre_vectors'][seeds].mean(axis=0,keepdims=True))
        content=dot_products(genre,self.a['genre_vectors'])
        scores=hybrid_scores(cf,content,.3)[0]
        results=self.rank(scores,seeds,k,diversity,seeds=seeds,**filters)
        return {'mode':'favorite_based_cold_start','model':self.config['best_learned_family'],
                'seconds':time.perf_counter()-start,'recommendations':results,
                'evaluation_note':'Favorite-based and diversity modes are not covered by warm-user benchmark metrics.'}

    def user(self,user_id,k=10,diversity=0.,**filters):
        if int(user_id) not in self.users: raise ValueError('Unknown benchmark profile')
        u=self.users[int(user_id)];name=self.config['recommended_model'];start=time.perf_counter()
        if name=='popularity': scores=self.a['popularity'].copy()
        elif name=='genre_content': scores=dot_products(self.a['genre_profiles'][[u]],self.a['genre_vectors'])[0]
        else:
            weight=float(name.split('_hybrid_')[1]) if '_hybrid_' in name else 0.
            scores=hybrid_scores(dot_products(self.a['user_vectors'][[u]],self.a['item_vectors']),
                                 dot_products(self.a['genre_profiles'][[u]],self.a['genre_vectors']),weight)[0]
        seeds=self.likes[u].indices[-5:].tolist()
        results=self.rank(scores,self.seen[u].indices,k,diversity,seeds=seeds,**filters)
        return {'mode':'existing_benchmark_user','model':name,'seconds':time.perf_counter()-start,'recommendations':results}


@lru_cache(maxsize=1)
def get_service(): return MovieService()
