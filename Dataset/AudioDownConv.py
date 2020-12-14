#imports
import requests
import pandas as pd
import os
from pydub import AudioSegment
import shutil

#url downloading
path = "mp3Data/"
def downloader(url, species, count):
    r = requests.get(url, allow_redirects=True)
    open(path + '{0}/{1}_{2}.mp3'.format(species, species, count), 'wb').write(r.content)

#Species download
def specDwnld():
    i = 1 #Initialise starting count
    for row in df.itertuples():
        if not os.path.exists(os.path.join(path,row[2])):
            os.makedirs(os.path.join(path,row[2]))
            i = 1
        downloader(row[1], row[2], i)
        i+=1

#convert mp3 to wav files
def wavConverter():
    audio_folds = os.listdir(path)
    for folder in audio_folds: 
        path1 = os.path.join(path,folder)
        audio_files = os.listdir(path1)
    
        if not os.path.exists(os.path.join('AudioFiles_wav',folder)):
            os.makedirs(os.path.join('AudioFiles_wav',folder))
    
        for file in audio_files:
            name, ext = os.path.splitext(file)
            mp3_sound = AudioSegment.from_file(os.path.join(path1,file))
            mp3_sound.export('AudioFiles_wav/{0}/{1}.wav'.format(folder, name), format="wav")
    shutil.rmtree(path)

if os.path.exists(path):    
    inp = input("mp3Data folder already exists. Proceed with conversion(y/n):")
    if (inp == "y"):
        print("Converting...")
        wavConverter()
        print("Conversion Complete")
else:
    csv_file = input("Enter csv filename to start download(e.g. AudioFilesTest.csv): ")
    df = pd.read_csv(csv_file)
    print("Downloading...")
    specDwnld()
    print("Download Complete")
    inp = input("Proceed with conversion(y/n):")
    if (inp == "y"):
        print("Converting...")
        wavConverter()
        print("Conversion Complete")