import pickle
import streamlit as st
import requests
from urllib.request import urlopen
import urllib
import json
import webbrowser
st.set_page_config(layout="wide")
if "load_state" not in st.session_state:
    st.session_state.load_state = False

def top_rated(i):
    url = "https://api.themoviedb.org/3/movie/top_rated?api_key=69c304f25f7dc5704a9e4d45bede72ff&language=en-US&page=1"
    data = requests.get(url)
    data = data.json()
    top = data['results'][i]['title']
    return top


def top_rated_images(i):
    url = "https://api.themoviedb.org/3/movie/top_rated?api_key=69c304f25f7dc5704a9e4d45bede72ff&language=en-US&page=1"
    data = requests.get(url)
    data = data.json()
    poster_path = data['results'][i]['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return full_path


def upcoming(i):
    url = "https://api.themoviedb.org/3/movie/upcoming?api_key=69c304f25f7dc5704a9e4d45bede72ff&language=en-US&page=1"
    data = requests.get(url)
    data = data.json()
    top = data['results'][i]['title']
    return top


def upcoming_images(i):
    url = "https://api.themoviedb.org/3/movie/upcoming?api_key=69c304f25f7dc5704a9e4d45bede72ff&language=en-US&page=1"
    data = requests.get(url)
    data = data.json()
    poster_path = data['results'][i]['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return full_path


def top_rated_overview(i):
    url = "https://api.themoviedb.org/3/movie/top_rated?api_key=776188a32a40d01731d846202af41221&language=en-US&page=1"
    data = requests.get(url)
    data = data.json()
    over = data['results'][i]['vote_average']
    return over


def popular_overview(i):
    url = "https://api.themoviedb.org/3/movie/popular?api_key=69c304f25f7dc5704a9e4d45bede72ff&language=en-US&page=1"
    data = requests.get(url)
    data = data.json()
    over = data['results'][i]['vote_average']
    return over


def popular_image(i):
    url = "https://api.themoviedb.org/3/movie/popular?api_key=69c304f25f7dc5704a9e4d45bede72ff&language=en-US&page=1"
    data = requests.get(url)
    pop = []
    data=data.json()
    # p=data['results'][i]['poster_path']
    poster_path=data['results'][i]['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return full_path


def popular(i):
    url = "https://api.themoviedb.org/3/movie/popular?api_key=69c304f25f7dc5704a9e4d45bede72ff&language=en-US&page=1"
    data=requests.get(url)
    pop = []
    data = data.json()
    p = data['results'][i]['original_title']
    # poster_path=data['results'][i]['poster_path']
    # full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return p


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=69c304f25f7dc5704a9e4d45bede72ff".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def fetch_watch_names(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}/watch/providers?api_key=69c304f25f7dc5704a9e4d45bede72ff".format(movie_id)
    response = urlopen(url)
    data_json = json.loads(response.read())
    if 'IN' in data_json['results'] and 'flatrate' in data_json['results']['IN']:
        op = (data_json['results']['IN']['flatrate'][0]['provider_name'])
        if op == "Hotstar":
            name = "Hotstar"
            return name
        elif op == "Amazon Prime Video" or op=="Amazon Video":
            name = "Amazon Prime"
            return name
        elif op == "Netflix":
            name = "Netflix"
            return name
        elif op == "Apple itunes" or op=="Apple TV":
            name="Apple TV"
            return name
        elif op == "Youtube":
            name="Youtube"
            return name
        elif op == "Google Play Movies":
            name = "Google Play Movies"
            return name
        elif op == "None":
            name = "Take me to google"
            return name
        else:
            return data_json['results']['IN']['flatrate'][0]['provider_name']
    return "Google it!"


def fetch_watch_providers(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}/watch/providers?api_key=69c304f25f7dc5704a9e4d45bede72ff".format(movie_id)
    response = urlopen(url)
    data_json = json.loads(response.read())
    if "load_state" not in st.session_state:
        st.session_state.load_state = False
    if 'IN' in data_json['results'] and 'flatrate' in data_json['results']['IN']:
        op=(data_json['results']['IN']['flatrate'][0]['provider_name'])
        if op == "Hotstar":
            link = "https://www.hotstar.com/in"
            return link
        elif op == "Amazon Prime Video" or op=="Amazon Video":
            link = "https://www.primevideo.com/"
            return link
        elif op == "Netflix":
            link = "https://www.netflix.com/in/"
            return link
        elif op == "Apple itunes" or op=="Apple TV":
            link = "https://tv.apple.com/"
            return link
        elif op == "Youtube":
            link = "https://www.youtube.com/"
            return link
        elif op == "Google Play Movies":
            link = "https://play.google.com/"
            return link
        else:
            link = "https://www.google.com/"
            return "https://www.google.com/"
    return "https://www.google.com/"

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_watch = []
    fetch_watch_names1 = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title.title())
        fetch_watch_names1.append(fetch_watch_names(movie_id))
        # fetch watch providers
        recommended_movie_watch.append(fetch_watch_providers(movie_id))
    return recommended_movie_names , recommended_movie_posters , recommended_movie_watch,fetch_watch_names1


# STREAMLIT CODE
st.title('Movie Recommender System')
st.caption("@The MovieDB")


movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
# recommender
st.subheader("Recommendations... ")
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation') or st.session_state.load_state:
    st.session_state.load_state = True
    recommended_movie_names, recommended_movie_posters, recommended_movie_watch, fetch_watch_names1 = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.subheader(recommended_movie_names[0])
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], use_column_width=True)
        # st.markdown(recommended_movie_watch[0])
        st.caption("Available on : ")
        l1 = recommended_movie_watch[0]
        a1 = fetch_watch_names1[0]
        if st.button(str(a1), key="1"):
            webbrowser.open(l1,autoraise=True)
            st.caption("If failed to open link...Try below ")
            st.markdown(l1, unsafe_allow_html=True)
            pass
    with col2:
        st.subheader(recommended_movie_names[1])
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], use_column_width=True)
        st.caption("Available on : ")
        l2 = recommended_movie_watch[1]
        a2 = fetch_watch_names1[1]
        if st.button(str(a2), key="2"):
            webbrowser.open(l2, autoraise=True)
            st.caption("If failed to open link...Try below ")
            st.markdown(l2, unsafe_allow_html=True)
            pass
    with col3:
        st.subheader(recommended_movie_names[2])
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], use_column_width =True)
        st.caption("Available on : ")
        l3 = recommended_movie_watch[2]
        a3 = fetch_watch_names1[2]
        if st.button(str(a3), key="3"):
            webbrowser.open(l3, autoraise=True)
            st.caption("If failed to open link...Try below ")
            st.markdown(l3, unsafe_allow_html=True)
            pass
    with col4:
        st.subheader(recommended_movie_names[3])
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], use_column_width=True)
        st.caption("Available on : ")
        l4 = recommended_movie_watch[3]
        a4 = fetch_watch_names1[3]
        if st.button(str(a4),key="4"):
            webbrowser.open(l4,autoraise=True)
            st.caption("If failed to open link...Try below ")
            st.markdown(l4, unsafe_allow_html=True)
            pass
    with col5:
        st.subheader(recommended_movie_names[4])
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], use_column_width =True)
        st.caption("Available on : ")
        l5 = recommended_movie_watch[4]
        a5 = fetch_watch_names1[4]
        if st.button(str(a5), key="5"):
            webbrowser.open(l5, autoraise=True)
            st.caption("If failed to open link...Try below ")
            st.markdown(l5, unsafe_allow_html=True)
            pass


st.header("")
st.header("")
st.header("")
st.subheader("Trending Today...")
# Trending
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
with c1:
    i = 1
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))
with c2:
    i = 2
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))
with c3:
    i = 3
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))
with c4:
    i = 4
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))
with c5:
    i = 5
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))
with c6:
    i = 6
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))
with c7:
    i = 7
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))
with c8:
    i = 8
    st.caption(popular(i))
    st.image(popular_image(i), popular_overview(i))


st.header("")
st.header("")
st.header("")
st.subheader("Top Rated...")
c01, c02, c03, c04, c05, c06, c07, c08 = st.columns(8)
with c01:
    i = 1
    st.caption(top_rated(i))
    st.image(top_rated_images(i), caption=top_rated_overview(i))
with c02:
    i = 2
    st.caption(top_rated(i))
    st.image(top_rated_images(i), caption=top_rated_overview(i))
with c03:
    i = 3
    st.caption(top_rated(i))
    st.image(top_rated_images(i), caption=top_rated_overview(i))
with c04:
    i = 4
    st.caption(top_rated(i))
    st.image(top_rated_images(i), caption=top_rated_overview(i))
with c05:
    i = 5
    st.caption(top_rated(i))
    st.image(top_rated_images(i), caption=top_rated_overview(i))
with c06:
    i = 6
    st.caption(top_rated(i))
    st.image(top_rated_images(i), caption=top_rated_overview(i))
with c07:
    i = 7
    st.caption(top_rated(i))
    st.image(top_rated_images(i),caption=top_rated_overview(i))
with c08:
    i = 8
    st.caption(top_rated(i))
    st.image(top_rated_images(i),caption=top_rated_overview(i))


st.header("")
st.header("")
st.header("")
st.subheader("Upcoming...")
c001, c002, c003, c004, c005, c006, c007, c008 = st.columns(8)
with c001:
    i = 1
    st.caption(upcoming(i))
    st.image(upcoming_images(i))
with c002:
    i = 2
    st.caption(upcoming(i))
    st.image(upcoming_images(i))
with c003:
    i = 3
    st.caption(upcoming(i))
    st.image(upcoming_images(i))
with c004:
    i = 4
    st.caption(upcoming(i))
    st.image(upcoming_images(i))
with c005:
    i = 5
    st.caption(upcoming(i))
    st.image(upcoming_images(i))
with c006:
    i = 6
    st.caption(upcoming(i))
    st.image(upcoming_images(i))
with c007:
    i = 7
    st.caption(upcoming(i))
    st.image(upcoming_images(i))
with c008:
    i = 8
    st.caption(upcoming(i))
    st.image(upcoming_images(i))


