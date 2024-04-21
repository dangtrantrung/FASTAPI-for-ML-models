import io
import pickle

import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# from sklearn.exceptions import InconsistentVersionWarning
# # warnings.simplefilter("error", InconsistentVersionWarning)

# try:
#    est = pickle.loads("models/mnist_model.pkl")
# except InconsistentVersionWarning as w:
#    print(w.original_sklearn_version)

with open('models/mnist_model.pkl', 'rb') as f:

    model= pickle.load(f)

app=FastAPI()
origins = ['http://127.0.0.1:5500',
'https://127.0.0.1:5500',
'http://127.0.0.1:8000/predict-image/',
'http://127.0.0.1:5500/APIs%20For%20ML%20Models%20with%20FastAPI/index.html']  # NOT recommended - see details below

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict-image/")
async def predict_image(file:UploadFile=File(...)):
    contents=await file.read()
    pil_image=PIL.Image.open(io.BytesIO(contents)).convert("L")
    pil_image=PIL.ImageOps.invert(pil_image)
    pil_image=pil_image.resize((28,28),PIL.Image.ANTIALIAS)
    img_array=np.array(pil_image).reshape(1,-1)
    prediction=model.predict(img_array)
    print(prediction)
    return {'predict': int(prediction[0])}

print("server API is starting")
# app.run(ssl_context='adhoc',host='0.0.0.0',port='8000')