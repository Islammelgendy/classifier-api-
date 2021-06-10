#import required packages
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import tensorflow as tf 
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from tensorflow import keras 
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.gen_math_ops import imag, mod 
import uvicorn 
from fastapi.responses import StreamingResponse,FileResponse
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from tensorflow import keras
from skimage.color import rgb2lab, deltaE_cie76
import os 
from scipy.spatial import KDTree
from webcolors import hex_to_rgb, hex_to_rgb_percent, rgb_to_name ,CSS3_HEX_TO_NAMES
from sklearn.cluster import KMeans
import tensorflow as tf 
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

#api initialization 
app =FastAPI(title='image classifier')

@app.get('/')
async def hello_world():
    return 'helloworld'
def create_model():
    model = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    classes=1000,
    classifier_activation="softmax")
    return model


@app.post('/product_tags /')
async def predict_tags(file:UploadFile=File(...)):
    contents=await file.read()
    im=Image.open(io.BytesIO(contents))
    #resizing images to (299,299) to fit int the model
    im =im.resize((299,299))
    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #load inception
    mod = create_model()
    preds = mod.predict(x)
    return str(decode_predictions(preds)[0])

    return 'mod'+ str (decode_predictions (preds, top=3 ))
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(image,number_of_colors,show_chart):
    modified_image = cv2.resize(image, (90, 90), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors ,n_init =20)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    return rgb_colors

def convert_rgb_to_names(rgb_tuple):
    import webcolors
    from webcolors import rgb_to_name ,CSS3_HEX_TO_NAMES 

    css3_db = webcolors.CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return  {names[index]}


@app.post('/product-color/')

async def predict_tags(file:UploadFile=File(...)):
    from tensorflow.keras.preprocessing import image
    l = []
    contents=await file.read()
    im=Image.open(io.BytesIO(contents))
    image = image.img_to_array(im)
    r = get_colors(image,3, 1)
    for i in range(len(r)):
        l.append(convert_rgb_to_names((r[i][0],(r[i][1]),( r[i][2]))))
    return str(l)

