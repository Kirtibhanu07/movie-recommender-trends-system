"""Run: .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8001"""
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from service import get_service

app=FastAPI(title='FrameLight Recommendation API',version='1.0.0')


class FavoriteRequest(BaseModel):
    movie_ids:list[int]=Field(min_length=1,max_length=20)
    k:int=Field(default=10,ge=1,le=50)
    diversity:float=Field(default=.2,ge=0,le=1)
    genres:list[str]|None=None
    year_min:int|None=None
    year_max:int|None=None


@app.get('/health')
def health():
    try:
        service=get_service()
        return {'status':'ready','model':service.config['recommended_model'],'catalogue_items':len(service.movies)}
    except FileNotFoundError:
        raise HTTPException(503,'Model artifacts are missing. Run train_evaluate.py first.')


@app.post('/recommend/favorites')
def recommend_favorites(request:FavoriteRequest):
    try: return get_service().favorites(**request.model_dump())
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    except FileNotFoundError: raise HTTPException(503,'Train the model first')


@app.get('/recommend/user/{user_id}')
def recommend_user(user_id:int,k:int=10):
    try: return get_service().user(user_id,k=k)
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    except FileNotFoundError: raise HTTPException(503,'Train the model first')
