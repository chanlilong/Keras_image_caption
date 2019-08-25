#%%
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import pandas as pd
import re
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet,InceptionV3
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
#%%
#============[LOAD WORDS DATAFRAME]=================#
words_df=pd.read_csv("words_list.csv")
#plt.bar(words_df["Word"].iloc[3:13],words_df["Counts"].iloc[3:13])
#============[MAKE DICTIONARY]===============#
index_to_words=dict(words_df["Word"])
index_to_words.pop(0,None)
words_to_index={y:x for x,y in index_to_words.items()}


#%%
def choose_img():
    global img,img2
    try:
        path = filedialog.askopenfilename(filetypes=[("Image File",".jpg .jpeg .gif .tmp .png")])
        #tkFileDialog.askopenfilename(filetypes=[("Image File",'.jpg')])
        # screen2=Toplevel(screen)
        # screen2.geometry("299x299")
        #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        img = (Image.open(path))
        #299X299 required for InceptionV3
        img=img.resize((299,299))
        img2=ImageTk.PhotoImage(img)
        panel.configure(image=img2)
        panel.image = img2
    except:
        caption.set("Please Choose an Image")
        pass
    # Label(screen2,image=img2).pack()

def initialise_model():
    global model,bottleneck_mdl
    from tensorflow.keras.layers import CuDNNGRU,Dropout,Dense,Embedding,CuDNNLSTM,Input,add

    #===========================================================#
    #=========================[MODEL]===========================#
    #===========================================================#
    bottleneck_input=Input(shape=(2048,))
    fe1=Dropout(0.5)(bottleneck_input)
    fe2=Dense(256,activation=tf.nn.selu)(fe1)
    
    #Partial Caption Input
    cap_inputs=Input(shape=(72,))
    #se1 is already pretrained on GLOVE model
    se1=Embedding(5411,200)(cap_inputs)
    se2=Dropout(0.5)(se1)
    se3=CuDNNGRU(256,return_sequences=True)(se2)
    se4=CuDNNLSTM(256)(se3)
    
    decoder1=add([fe2,se4])
    decoder2=Dense(256,activation=tf.nn.selu)(decoder1)
    outputs=Dense(5411,activation=tf.nn.softmax)(decoder2)
    
    model=Model(inputs=[bottleneck_input,cap_inputs],outputs=outputs)

    model.load_weights('model_xeon.h5')
    #===========================================================#
    #=========================[MODEL]===========================#
    #===========================================================#

    incep_mdl=InceptionV3(weights="imagenet")
    bottleneck_mdl=Model(incep_mdl.input,incep_mdl.layers[-2].output)

def create_bottleneck(image):
    image=np.asarray(image)
    x=np.expand_dims(image,axis=0)
    x=preprocess_input(x)
    x=bottleneck_mdl.predict(x)
    return(x)

def run_model():
    global final
    # del_prev()
    bottleneck=create_bottleneck(img)
    max_cap_length=72
    in_text = 'startseq'
    for i in range(max_cap_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        temp=np.zeros(max_cap_length-len(sequence))
        dab=np.concatenate([sequence,temp])
        dab=np.asarray(dab,dtype=np.int16)
        yhat = model.predict([bottleneck.reshape(1,2048),dab.reshape(1,max_cap_length)], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    caption.set(final)
#%%
initialise_model()
#%%
def main_scr():
    global screen,caption,panel
    screen=Tk()
    screen.geometry("1280x720")
    screen.title("KERAS Image Captioner v1.0")
    Label(text="Author: Chan Li Long\nDated: Aug 2019",bg="grey",font=("Calibri",13)).pack()
    Label(text="").pack()
    Label(text="KERAS IMAGE CAPTION V1.0",fg="red",bg="grey",font=("Calibri",13)).pack()
    Label(text="").pack()
    img=(Image.open("test.jpg"))
    img=img.resize((299,299))
    img=ImageTk.PhotoImage(img)
    panel = Label(screen, image=img)
    panel.pack()
    Button(text="Choose Image",command=choose_img).pack()
    Label(text="").pack()
    caption=StringVar()
    caption.set("Caption Will Be Shown Here")
    Label(textvariable=caption,bg="white").pack()
    Button(text="CAPTION!",fg="green",command=run_model,height=5,width=15).pack()
    
    screen.mainloop()

main_scr()
