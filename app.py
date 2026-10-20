"""FrameLight, a local movie discovery app backed by trained collaborative embeddings."""
import html
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from service import get_service
from recommender import ROOT

st.set_page_config(page_title='FrameLight | Movie discovery',page_icon='🎬',layout='wide')
st.markdown('''<style>
.block-container{max-width:1180px;padding-top:2rem}
.hero{padding:30px 32px;border:1px solid #3d315f;border-radius:18px;background:linear-gradient(125deg,#201836,#14263b);margin-bottom:24px}
.eyebrow{font-size:12px;letter-spacing:3px;color:#c7afff;text-transform:uppercase}
.hero h1{font-size:46px;letter-spacing:-1.8px;margin:8px 0;color:#fff}
.hero p{color:#c4c7d4;font-size:17px;max-width:650px}
.movie{min-height:200px;background:#181b2b;border:1px solid #33364a;border-radius:14px;padding:20px;margin-bottom:16px}
.number{font-size:12px;color:#b6a0ff;letter-spacing:2px}
.movie h3{font-size:20px;color:#f4f3fa;margin:10px 0}
.genre{color:#e6b56b;font-size:12px}.reason{color:#b6bbce;font-size:13px;margin-top:12px}
</style>''',unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">FrameLight / Movie discovery</div><h1>Find your next favorite.</h1><p>Start with films you love. Explore a collection of classics through learned preferences and a little curiosity.</p></div>''',unsafe_allow_html=True)
try: service=get_service()
except FileNotFoundError:
    st.info('Run download_data.py and train_evaluate.py to prepare the local model.');st.stop()

discover,profiles,benchmark=st.tabs(['Discover','Example profiles','Model benchmark'])


def cards(response):
    if not response['recommendations']: st.info('No movies match these filters. Try a wider year range or genre.');return
    st.caption(f"Ranked locally in {response['seconds']*1000:.1f} ms. Scores describe ranking, not the probability you will like a film.")
    columns=st.columns(3)
    for index,movie in enumerate(response['recommendations']):
        with columns[index%3]:
            st.markdown(f'''<div class="movie"><div class="number">PICK {movie['rank']:02d} / {movie['year']}</div><h3>{html.escape(movie['title'])}</h3><div class="genre">{html.escape(' · '.join(movie['genres']))}</div><div class="reason">{html.escape(movie['reason'])}</div></div>''',unsafe_allow_html=True)


with discover:
    left,right=st.columns([2,1])
    labels={row.title:int(row.movie_id) for row in service.movies.itertuples()}
    default=[t for t in ['Matrix, The (1999)','Toy Story (1995)','Shawshank Redemption, The (1994)'] if t in labels]
    with left:
        favorites=st.multiselect('Choose a few films you love',list(labels),default=default,max_selections=20)
    with right:
        diversity=st.slider('Familiar to adventurous',0.,1.,.25,.05)
    with st.expander('Fine-tune your picks'):
        all_genres=sorted({g for s in service.movies.genres for g in s.split('|')})
        genres=st.multiselect('Genres to include',all_genres)
        years=st.slider('Release years',int(service.movies.year.min()),int(service.movies.year.max()),
                        (int(service.movies.year.min()),int(service.movies.year.max())))
    if favorites:
        cards(service.favorites([labels[t] for t in favorites],k=12,diversity=diversity,
                                genres=genres or None,year_min=years[0],year_max=years[1]))
    else: st.info('Choose at least one movie to begin.')
    st.caption('This is a historical MovieLens catalogue. Favorite-based discovery and diversity controls have not been evaluated as live personalization.')

with profiles:
    st.subheader('See a learned profile in action')
    st.write('These anonymous benchmark profiles use the trained user embeddings. Previously rated films are excluded.')
    eligible=service.a['eligible_users']
    choices=[int(service.a['user_ids'][u]) for u in eligible[:20]]
    user=st.selectbox('Example profile',choices,format_func=lambda x:f'Benchmark profile {x}')
    cards(service.user(user,k=12))

with benchmark:
    st.subheader('Measured, not guessed')
    selection=json.loads((ROOT/'evaluation_results/selection.json').read_text())
    metrics=pd.read_csv(ROOT/'evaluation_results/test_metrics.csv')
    winner=metrics[metrics.model==selection['recommended_model']].iloc[0]
    cols=st.columns(3)
    cols[0].metric('NDCG@20',f"{winner['NDCG@20']:.3f}")
    cols[1].metric('Recall@20',f"{winner['Recall@20']:.1%}")
    cols[2].metric('Eligible test users',f"{int(winner.users):,}")
    st.dataframe(metrics[['model','NDCG@20','Recall@20','catalogue_coverage@20']],hide_index=True,use_container_width=True)
    st.caption('Per-user chronological holdout with full-catalogue ranking. This is not a global-time simulation or an online A/B test. These warm-user metrics do not measure the favorite-based discovery mode.')
    st.write('LightGCN and BPR matrix factorization are compared with popularity and content baselines. Model and checkpoint selection use validation data only.')

st.caption('Research demonstration powered by MovieLens 1M. Dataset use is subject to GroupLens terms. No commercial deployment or endorsement is claimed.')
