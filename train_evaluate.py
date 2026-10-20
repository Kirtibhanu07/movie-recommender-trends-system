"""Train on MovieLens 1M, select on validation, score the untouched test split."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from recommender import (ROOT, GraphRecommender, load_dataset, chronological_split,
                         positive_matrix, seen_matrix, genre_matrix, normalized_graph,
                         sample_triplets, evaluate_scores, content_profile, hybrid_scores,
                         paired_bootstrap, dot_products)


def export_vectors(model,graph):
    with torch.inference_mode():
        vectors=model.propagate(graph).cpu().numpy()
    return vectors[:model.n_users].copy(),vectors[model.n_users:].copy()


def train_model(name,layers,plan,positives,seen,truth,val_users,graph,out):
    torch.manual_seed(plan['random_seed']);rng=np.random.default_rng(plan['random_seed'])
    model=GraphRecommender(*positives.shape,plan['embedding_dimension'],layers)
    optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
    best=-float('inf');best_vectors=None;history=[];started=time.perf_counter()
    for epoch in range(1,plan['max_epochs']+1):
        losses=[]
        for _ in range(plan['updates_per_epoch']):
            batch=sample_triplets(positives,plan['triples_per_update'],rng)
            optimizer.zero_grad(set_to_none=True)
            loss=model.bpr_loss(graph,*batch,l2=plan['l2'])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        if epoch%plan['validation_every']==0:
            u,i=export_vectors(model,graph)
            metric,_=evaluate_scores(name,val_users,lambda users:dot_products(u[users],i),seen,truth)
            entry={'model':name,'epoch':epoch,'BPR_loss':float(np.mean(losses)),
                   'validation_NDCG20':metric['NDCG@20'],'elapsed_seconds':time.perf_counter()-started}
            history.append(entry);print(entry,flush=True)
            if metric['NDCG@20']>best:
                best=metric['NDCG@20'];best_vectors=(u,i,epoch)
                # Safe, tensor-only state, no arbitrary pickled custom classes.
                torch.save(model.state_dict(),out/f'{name}_checkpoint.pt')
    return best_vectors,history


def save_report(out,summary,validation,test,inference,latency,selection):
    selected=selection['recommended_model']
    best=test.set_index('model').loc[selected];base=test.set_index('model').loc['popularity']
    lift=100*(best['NDCG@20']/base['NDCG@20']-1)
    table='| Model | NDCG@20 | Recall@20 | Precision@20 | Catalogue coverage |\n|---|---:|---:|---:|---:|\n'
    for _,r in test.iterrows():
        table+=f"| {r.model} | {r['NDCG@20']:.4f} | {r['Recall@20']:.2%} | {r['Precision@20']:.2%} | {r['catalogue_coverage@20']:.2%} |\n"
    ci=inference['95pct_paired_user_bootstrap_interval']
    text=f"""# Personalized movie recommendation benchmark

The validation-selected recommendation is **{selected}**. On {int(best.users)} eligible
test users it achieved **NDCG@20 {best['NDCG@20']:.4f}** and **Recall@20 {best['Recall@20']:.2%}**.
Its NDCG@20 is {lift:.1f}% higher than the popularity baseline on this split.
This is a retrospective benchmark, not online engagement or commercial impact.

## Dataset and split

Official MovieLens 1M has {summary['ratings']:,} ratings, {summary['users']:,} users and
{summary['catalogue_items']:,} catalogue items. A rating of 4 or 5 is treated as a positive.
Only movie metadata and ratings are used. Demographic data is not loaded.

Distinct timestamp groups are ordered within each user. The earliest 70% of groups
are training, the next 15% validation, and the remaining groups test. Tied timestamp
groups never cross boundaries. Users with fewer than three groups remain training-only.
Eligible test users need five training positives and at least one positive in both
validation and test. This selects a particular population and is not representative
of every user. A seeded subset of {summary['validation_users']} eligible users is used
for checkpoint and blending selection. Final testing uses all eligible users.

This is **per-user chronological evaluation**, not a globally chronological simulation.
Another user's training event may occur later than a query user's held-out event.
The catalogue is assumed static. Ratings do not establish movie watch order, exposure,
or satisfaction. Unobserved movies are not known dislikes.

## Models and selection

- Popularity from training positive counts.
- Content profiles from training likes and IDF-weighted movie genres.
- Bayesian Personalized Ranking matrix factorization, 32-dimensional embeddings.
- Two-layer LightGCN with normalized user-item graph propagation and BPR loss.
- A validation-selected genre blend for the better learned checkpoint.

Both learned models use the same fixed training budget and seed. The model graph,
negative rejection and content profiles use training ratings only. Epochs and genre
weights are selected using validation NDCG@20. All choices are saved before scoring
the final test. No refit or test-based retuning is performed. Equal training budgets
do not establish a globally optimized comparison or state-of-the-art result.

Test ranking considers the full {summary['catalogue_items']:,}-item catalogue, removing
items already rated in training or validation. There is no sampled-negative ranking.
Metrics are macro averages over users. Recall@20 measures the fraction of the user's
held-out positive movies retrieved, not classification accuracy.

## Final test

{table}
## Uncertainty and serving measurements

The selected-minus-popularity NDCG@20 difference is
{inference['mean_NDCG20_difference']:.4f}, with a 95% paired user-bootstrap interval
of [{ci[0]:.4f}, {ci[1]:.4f}]. This measures user sampling uncertainty on one split
and one training seed. It does not include training variability or online effects.

Warm full-catalogue ranking median is {latency['median_ms']:.2f} ms and p95 is
{latency['p95_ms']:.2f} ms across {latency['queries']} recorded queries. This includes
score calculation, filtering and sorting but not app rendering, network or model load.
It is a local single-process benchmark, not a production throughput test.

## Reproducibility and distribution

See the frozen experiment plan, training history, validation leaderboard,
`selection.json`, test leaderboard, per-user derived metrics and environment/data
hashes. Run `download_data.py`, then `train_evaluate.py` to reproduce. The raw dataset
and data-derived serving artifacts are not included in shared source archives.

MovieLens 1M is research-only and disallows raw data redistribution without separate
permission. This demonstration is not licensed for commercial deployment. Cite:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
Context. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872

LightGCN follows Xiangnan He et al., SIGIR 2020: https://arxiv.org/abs/2002.02126

## Limits

The data is from around 2000 and does not measure recommendations for today's
catalogue or current audiences. There is one seed, a limited hyperparameter budget,
and no online A/B test. Cold-start favorite-based recommendations and diversity
reranking are separate app features, not covered by the warm-user metrics above.
The trained model is new to this project, not a novel research architecture.
"""
    (out/'REPORT.md').write_text(text)
    (out/'resume_evidence.txt').write_text(
        f"Built a personalized movie recommender using MovieLens 1M with {summary['ratings']:,} ratings. "
        f"Trained BPR matrix factorization and LightGCN with per-user chronological validation. "
        f"The validation-selected {selected} achieved NDCG@20 {best['NDCG@20']:.3f} and Recall@20 {best['Recall@20']:.1%} "
        f"across {int(best.users):,} eligible test users, with full-catalogue ranking and paired bootstrap uncertainty estimates.\n")


def plot_results(out,history,test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name,g in history.groupby('model'):
        axes[0].plot(g.epoch,g.validation_NDCG20,marker='o',label=name)
    axes[0].set(title='Validation learning curves',xlabel='Epoch',ylabel='NDCG@20');axes[0].legend()
    axes[1].barh(test.model,test['NDCG@20'],color='#8b5cf6');axes[1].invert_yaxis()
    axes[1].set(title='Held-out test, full-catalogue ranking',xlabel='NDCG@20')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)


def main():
    plan=json.loads((ROOT/'experiment_plan.json').read_text());torch.set_num_threads(plan['cpu_threads'])
    torch.manual_seed(plan['random_seed']);np.random.seed(plan['random_seed'])
    out=ROOT/'evaluation_results';out.mkdir(exist_ok=True)
    artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True)
    ratings,movies,user_ids=load_dataset();split=chronological_split(ratings)
    n_users,n_items=len(user_ids),len(movies)
    train=split[split.split=='train'];val=split[split.split=='validation'];test=split[split.split=='test']
    positives=positive_matrix(train,n_users,n_items);seen=seen_matrix(train,n_users,n_items)
    val_truth=positive_matrix(val,n_users,n_items);test_truth=positive_matrix(test,n_users,n_items)
    counts=np.asarray(positives.sum(1)).ravel()
    eligible=np.flatnonzero((counts>=5)&(np.asarray(val_truth.sum(1)).ravel()>=1)&(np.asarray(test_truth.sum(1)).ravel()>=1))
    if len(eligible)<100: raise ValueError('Too few eligible users for this experiment')
    rng=np.random.default_rng(42);validation_users=np.sort(rng.choice(eligible,min(1024,len(eligible)),replace=False))
    genres,genre_names=genre_matrix(movies);profiles=content_profile(positives,genres)
    popularity=np.asarray(positives.sum(0)).ravel().astype(np.float32)
    graph=normalized_graph(positives)
    summary={'ratings':len(ratings),'users':n_users,'catalogue_items':n_items,'eligible_users':len(eligible),
             'validation_users':len(validation_users),'train_ratings':len(train),'train_positives':int(positives.sum()),
             'validation_ratings':len(val),'test_ratings':len(test),
             'dataset_sha256':{p:hashlib.sha256((ROOT/'data'/p).read_bytes()).hexdigest() for p in ['ratings.dat','movies.dat']}}
    (out/'data_audit.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
    histories=[];vectors={};validation=[];score_functions={
        'popularity':lambda users:np.broadcast_to(popularity,(len(users),n_items)),
        'genre_content':lambda users:dot_products(profiles[users],genres)}
    for name,fn in score_functions.items():
        metric,_=evaluate_scores(name,validation_users,fn,seen,val_truth);validation.append(metric)
    for name,layers in [('BPR_MF',0),('LightGCN',plan['lightgcn_layers'])]:
        best,history=train_model(name,layers,plan,positives,seen,val_truth,validation_users,graph,artifacts)
        histories.extend(history);u,i,epoch=best;vectors[name]=(u,i)
        score_functions[name]=lambda users,u=u,i=i:dot_products(u[users],i)
        metric,_=evaluate_scores(name,validation_users,score_functions[name],seen,val_truth)
        metric['epoch']=epoch;validation.append(metric)
    better=max([r for r in validation if r['model'] in vectors],key=lambda r:r['NDCG@20'])['model']
    vu,vi=vectors[better]
    for weight in plan['hybrid_genre_weights'][1:]:
        name=f'{better}_hybrid_{weight}'
        score_functions[name]=lambda users,w=weight:hybrid_scores(dot_products(vu[users],vi),dot_products(profiles[users],genres),w)
        metric,_=evaluate_scores(name,validation_users,score_functions[name],seen,val_truth);validation.append(metric)
    validation=pd.DataFrame(validation).sort_values(['NDCG@20','model'],ascending=[False,True])
    selected=str(validation.iloc[0].model)
    selection={'recommended_model':selected,'best_learned_family':better,'validation_metric':'NDCG@20',
               'best_learned_epochs':{r['model']:int(r['epoch']) for r in validation.to_dict('records') if r.get('model') in vectors},
               'validation_users':len(validation_users),'test_users':len(eligible)}
    # All checkpoint/model/blend choices locked before reading final-test labels for scoring.
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    validation.to_csv(out/'validation_metrics.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False)
    print('Locked recommendation:',selected,flush=True)
    final_seen=seen_matrix(pd.concat([train,val]),n_users,n_items)
    final_rows=[];per_user=[]
    # Keep one hybrid chosen only on validation, not all blends on the final test.
    hybrid_names=[n for n in score_functions if '_hybrid_' in n]
    best_hybrid=max(hybrid_names,key=lambda n:float(validation[validation.model==n]['NDCG@20'].iloc[0]))
    finalists=['popularity','genre_content','BPR_MF','LightGCN',best_hybrid]
    if selected not in finalists: finalists.append(selected)
    for name in finalists:
        metric,details=evaluate_scores(name,eligible,score_functions[name],final_seen,test_truth)
        final_rows.append(metric);per_user.append(details);print('Test',metric,flush=True)
    final=pd.DataFrame(final_rows).sort_values('NDCG@20',ascending=False)
    all_users=pd.concat(per_user,ignore_index=True)
    inference=paired_bootstrap(all_users[all_users.model==selected],all_users[all_users.model=='popularity'])
    timings=[]
    for user in np.resize(eligible,200):
        start=time.perf_counter();scores=np.asarray(score_functions[selected](np.array([user])))[0].copy()
        scores[final_seen[user].indices]=-np.inf;np.argsort(-scores,kind='stable')[:20]
        timings.append((time.perf_counter()-start)*1000)
    latency={'queries':len(timings),'median_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95))}
    final.to_csv(out/'test_metrics.csv',index=False);all_users.to_csv(out/'test_user_metrics.csv',index=False)
    (out/'inference.json').write_text(json.dumps(inference,indent=2));(out/'latency.json').write_text(json.dumps(latency,indent=2))
    # Serve compact learned embeddings, not an all-pairs dense similarity pickle.
    np.savez_compressed(artifacts/'serving.npz',user_vectors=vu,item_vectors=vi,genre_vectors=genres,
                        genre_profiles=profiles,popularity=popularity,user_ids=user_ids,
                        movie_ids=movies.movie_id.to_numpy(),eligible_users=eligible)
    sparse.save_npz(artifacts/'seen.npz',final_seen);sparse.save_npz(artifacts/'training_likes.npz',positives)
    movies.to_csv(artifacts/'catalogue.csv',index=False)
    (artifacts/'model_config.json').write_text(json.dumps(selection,indent=2))
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','torch','scikit-learn']},
                 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'core_sha256':hashlib.sha256((ROOT/'recommender.py').read_bytes()).hexdigest(),
                 'plan_sha256':hashlib.sha256((ROOT/'experiment_plan.json').read_bytes()).hexdigest()}
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    save_report(out,summary,validation,final,inference,latency,selection)
    plot_results(out,pd.DataFrame(histories),final)
    print('Complete. Serving embeddings and benchmark report written.',flush=True)


if __name__=='__main__': main()
