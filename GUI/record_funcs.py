# -*- coding: utf-8 -*-


import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import DataAug_misc as da
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from collections import Counter

#print(sd.query_devices())
#print(sd.default.device)

sr = 22050  # Sample rate
seconds = 5  # Duration of recording

model = keras.models.load_model('model_save/4_m2v3_12_Adam.h5')
labels={0:'Noise',1:'Corvus corax',2:'Cuculus canorus',3:'Parus major',4:'Passer domesticus'}


def plot_mfcc(mfccs):
  mfcc=np.array(mfccs)
  mfcc=np.squeeze(mfccs)
  print(mfcc.shape)
  plt.figure(figsize=(10,5))
  librosa.display.specshow(mfcc,x_axis='log',sr=sr)
  plt.show()
  
  
  
def record_py(seconds):
    wav_arr = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()  # Wait until recording is finished
    wav_arr=np.squeeze(wav_arr)
    print(wav_arr.shape)
    return upload_pred(wav_arr)
  

#split into mfccs as make predictions
def upload_pred(sigwav):
      mfccs=[]
      wavs = da.wav_split(sigwav)
      i=1
      for wav in wavs:
        mfcc = da.wav_to_mfcc(wav)
        #print(mfcc)
        print("Clip..{}".format(i))
        plot_mfcc(mfcc)
        mfccs.append(mfcc)
        mfcc=mfcc[np.newaxis,...]
        print(mfcc.shape)
        label=model.predict_classes(mfcc)
        print("Class predicted..."+labels[label[0]])
        i+=1
      
      mfccs = np.asarray(mfccs)
      print(mfccs.shape)
      print(model.predict(mfccs))
      preds =np.argmax(model.predict(mfccs),axis=-1)
      return preds,len(wavs)

#find the overall prediction
def predict(preds,wavs):
    correct=[]
    print(preds)
    for pred in preds:
        class_name=labels[pred]
        correct.append(class_name)
    
    print(len(correct))
    print(Counter(correct).most_common(1))
    f_pred=Counter(correct).most_common(1)[0][0]
    l_pred=Counter(preds).most_common(1)[0][0]
    print("Total no.of clips predicted...{}".format(wavs))
    print(Counter(correct))
    print("Predicted bird is..."+f_pred)
    return correct,f_pred,l_pred

#to play recording
def play_rec(sig,sr):
    sd.play(sig,sr)
    status=sd.wait()
        