import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch
from recommender import (ROOT,GraphRecommender,chronological_split,normalized_graph,
                         positive_matrix,sample_triplets,ranking_metrics,ranked_topk,
                         evaluate_scores,dot_products,hybrid_scores)


def test_timestamp_groups_never_cross_splits():
    rows=[]
    for u in [0,1]:
        for t in range(10):
            for i in [t*2,t*2+1]: rows.append({'u':u,'i':i,'rating':5,'timestamp':t})
    result=chronological_split(pd.DataFrame(rows))
    assert result.groupby(['u','timestamp']).split.nunique().eq(1).all()
    for _,g in result.groupby('u'):
        assert g[g.split=='train'].timestamp.max()<g[g.split=='validation'].timestamp.min()
        assert g[g.split=='validation'].timestamp.max()<g[g.split=='test'].timestamp.min()


def test_single_session_user_is_training_only():
    f=pd.DataFrame({'u':[0,0],'i':[1,2],'rating':[4,5],'timestamp':[10,10]})
    assert chronological_split(f).split.eq('train').all()


def test_graph_is_symmetric_and_handles_isolated_items():
    pos=sparse.csr_matrix(np.array([[1,0,0],[0,1,0]],np.float32))
    graph=normalized_graph(pos).to_dense().numpy()
    np.testing.assert_allclose(graph,graph.T)
    assert np.isfinite(graph).all() and np.all(graph[4]==0)
    assert graph[0,2]>0 and graph[0,3]==0


def test_positives_exclude_low_ratings():
    f=pd.DataFrame({'u':[0,0],'i':[0,1],'rating':[3,4]})
    p=positive_matrix(f,1,2)
    assert p[0,0]==0 and p[0,1]==1


def test_negatives_never_include_training_positive_edges():
    p=sparse.csr_matrix(np.array([[1,1,0,0],[0,0,1,1]],np.float32))
    u,pos,neg=sample_triplets(p,1000,np.random.default_rng(42))
    assert np.all(np.asarray(p[u.numpy().copy(),pos.numpy().copy()]).ravel()==1)
    assert np.all(np.asarray(p[u.numpy().copy(),neg.numpy().copy()]).ravel()==0)


def test_negative_sampling_rejects_full_catalogue_user():
    with pytest.raises(ValueError): sample_triplets(sparse.csr_matrix(np.ones((1,3))),10,np.random.default_rng(1))


def test_bpr_optimization_reduces_loss_and_has_finite_gradients():
    torch.manual_seed(42);torch.set_num_threads(1)
    p=sparse.csr_matrix(np.array([[1,1,0,0],[0,0,1,1]],np.float32));graph=normalized_graph(p)
    model=GraphRecommender(2,4,8,2);opt=torch.optim.Adam(model.parameters(),lr=.05)
    batch=sample_triplets(p,64,np.random.default_rng(42));losses=[]
    for _ in range(30):
        opt.zero_grad();loss=model.bpr_loss(graph,*batch);loss.backward()
        assert torch.isfinite(model.embedding.weight.grad).all()
        opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<.3*losses[0]


def test_exact_ranking_metric_definitions():
    m=ranking_metrics([1,2,3,4,5],{1,2})
    assert m['NDCG@20']==pytest.approx(1)
    assert m['Recall@20']==1 and m['Precision@20']==.1 and m['HitRate@20']==1
    assert ranking_metrics(list(range(20)),{99})['NDCG@20']==0
    with pytest.raises(ValueError): ranking_metrics([1],set())


def test_deterministic_tie_breaking():
    assert ranked_topk(np.ones(30),3).tolist()==[0,1,2]


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError): dot_products(np.array([[np.nan]],np.float32),np.ones((2,1),np.float32))


def test_scoring_backend_matches_double_precision_reference():
    rng=np.random.default_rng(42);a=rng.normal(size=(3,32)).astype(np.float32);b=rng.normal(size=(80,32)).astype(np.float32)
    np.testing.assert_allclose(dot_products(a,b),np.einsum('ud,id->ui',a.astype(float),b.astype(float)),rtol=2e-5,atol=1e-5)


def test_seen_items_cannot_enter_topk():
    scores=np.arange(30,dtype=np.float32)
    seen=sparse.csr_matrix(([1],([0],[29])),shape=(1,30))
    truth=sparse.csr_matrix(([1],([0],[28])),shape=(1,30))
    metric,_=evaluate_scores('x',np.array([0]),lambda u:np.array([scores]),seen,truth)
    assert metric['NDCG@20']==1


def test_hybrid_zero_weight_preserves_original_scores():
    a=np.array([[1.,2.,3.]],np.float32)
    np.testing.assert_array_equal(hybrid_scores(a,np.ones_like(a),0),a)


@pytest.fixture
def service():
    if not (ROOT/'artifacts/serving.npz').exists(): pytest.skip('Run train_evaluate.py for integration tests')
    from service import MovieService
    return MovieService()


def test_favorite_mode_excludes_seeds_and_returns_unique_movies(service):
    ids=service.movies.movie_id.iloc[:3].tolist()
    r=service.favorites(ids,k=12,diversity=.5)
    recommended=[m['movie_id'] for m in r['recommendations']]
    assert len(recommended)==len(set(recommended))==12
    assert not set(ids)&set(recommended)


def test_recommendation_filters_and_bad_ids(service):
    seed=int(service.movies.movie_id.iloc[0])
    r=service.favorites([seed],k=10,genres=['Drama'],year_min=1990,year_max=2000)
    assert all('Drama' in m['genres'] and 1990<=m['year']<=2000 for m in r['recommendations'])
    with pytest.raises(ValueError): service.favorites([-1])
    with pytest.raises(ValueError): service.favorites([seed],k=0)
    with pytest.raises(ValueError): service.favorites([seed],genres=['invalid'])
    with pytest.raises(ValueError): service.favorites([seed],year_min=2000,year_max=1900)


def test_warm_user_excludes_every_previously_rated_item(service):
    u=int(service.a['eligible_users'][0]);uid=int(service.a['user_ids'][u])
    result=service.user(uid,k=20)
    seen_ids=set(service.movies.iloc[service.seen[u].indices].movie_id)
    assert not seen_ids&{m['movie_id'] for m in result['recommendations']}


def test_saved_model_was_selected_on_validation_only():
    out=ROOT/'evaluation_results'
    if not (out/'selection.json').exists(): pytest.skip('Run training first')
    selection=json.loads((out/'selection.json').read_text());val=pd.read_csv(out/'validation_metrics.csv')
    assert selection['recommended_model']==val.sort_values(['NDCG@20','model'],ascending=[False,True]).iloc[0].model
    if not (out/'test_user_metrics.csv').exists():
        return  # Per-user diagnostics are generated locally by training.
    per=pd.read_csv(out/'test_user_metrics.csv');scores=pd.read_csv(out/'test_metrics.csv')
    assert per.groupby('model').user_index.nunique().eq(selection['test_users']).all()
    for row in scores.itertuples(index=False):
        frame=per[per.model==row.model]
        for metric in ['NDCG@20','Recall@20','Precision@20']:
            assert frame[metric].mean()==pytest.approx(float(scores[scores.model==row.model][metric].iloc[0]))


def test_api_validation_and_live_recommendations(service):
    from fastapi.testclient import TestClient
    from api import app
    client=TestClient(app)
    assert client.get('/health').status_code==200
    assert client.post('/recommend/favorites',json={'movie_ids':[]}).status_code==422
    assert client.post('/recommend/favorites',json={'movie_ids':[-1]}).status_code==400
    response=client.post('/recommend/favorites',json={'movie_ids':[int(service.movies.movie_id.iloc[0])],'k':5})
    assert response.status_code==200 and len(response.json()['recommendations'])==5


def test_streamlit_app_renders_without_errors():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    if (ROOT/'artifacts/serving.npz').exists():
        assert len(app.tabs)==3
        app.multiselect[0].set_value([]).run()
        assert not app.exception
