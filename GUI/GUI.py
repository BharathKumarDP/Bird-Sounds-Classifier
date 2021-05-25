# -*- coding: utf-8 -*-
import tkinter as tk
import librosa
from tkinter.filedialog import askopenfilename
import record_funcs as rec
from tkinter import ttk
from collections import Counter
from PIL import ImageTk, Image
import webbrowser
import tensorflow.keras as keras

img_path={0:"noise.jpg",1:"bird1.jpg",2:"bird2.jpg",3:"bird4.jpg",4:"bird5.jpg"}
wiki_path={0:"White_noise",1:"Corvus_corax",2:"Cuculus_canorus",3:"Parus_major",4:"Passer_domesticus"}

#Result window
def result(correct,bird,lbl):
    root= tk.Toplevel()
    root.title(bird)  
           
    canvas =tk.Canvas(root, width = 300, height = 300)      
    canvas.pack()      
    global img
    img= ImageTk.PhotoImage(Image.open(img_path[lbl]))      
    canvas.create_image(20,20, anchor=tk.NW, image=img)
    canvas.image=img      
    
    T =tk.Text(root, height = 5, width = 52)
      
    l =tk.Label(root, text = "Prediction results")
    l.config(font =("Courier", 14))
      
    tot=str(len(correct))
    txt1 = """Total number of clips predicted is..."""+tot
    txt2 = """\nCounter.."""
    txt3 = Counter(correct)
    txt4 = """\nPredicted bird is.."""+bird
    
    b1 = tk.Button(root, text = "More info" ,command=(lambda: link(lbl)))
    b2 = tk.Button(root, text = "Exit",
                command = root.destroy) 
    l.pack()
    T.pack()
    b1.pack()
    b2.pack()
      
    T.insert(tk.END, txt1)
    T.insert(tk.END, txt2)
    T.insert(tk.END, txt3)
    T.insert(tk.END, txt4)
    
    root.mainloop()

#Upload recording function
def upload(ent_upload):
    filepath=ent_upload.get()
    print(filepath)
    if not filepath:
        filepath = askopenfilename(
            filetypes=[("wav files", "*.wav"), ("All Files", "*.*")]
        )
    if not filepath:
        return
    sig,sr=librosa.load(filepath)
    dur=int(librosa.get_duration(sig,sr))
    win=tk.Toplevel(relief=tk.SUNKEN,borderwidth=2,height=200,width=300)
    win.title("Playing Audio")
    con=tk.Label(win,text="Uploaded Audio is being played....")
    con.pack()
    win.after(dur,lambda:win.destroy())
    rec.play_rec(sig,sr)
    preds,wavs=rec.upload_pred(sig)
    correct,bird,lbl=rec.predict(preds,wavs)
    result(correct,bird,lbl)
    return 

#Upload model function
def m_upload(ent_model):
    filepath=ent_model.get()
    print(filepath)
    if not filepath:
        filepath = askopenfilename(
            filetypes=[("h5 files", "*.h5"), ("All Files", "*.*")]
        )
    if not filepath:
        return
    model=keras.models.load_model(filepath)
    return 

#Record function
def record(ent_record):
      seconds=int(ent_record.get())
      preds,wavs=rec.record_py(seconds)
      correct,bird,lbl=rec.predict(preds,wavs)
      result(correct,bird,lbl)
      return

#About window
def abt():
    win=tk.Toplevel(relief=tk.SUNKEN,borderwidth=2,height=200,width=300)
    win.title("About")
    con=tk.Label(win,text="This is a tool to predict Birds from their Sounds")
    con.pack()
    con2=tk.Label(win,text="Upload-to upload a wav file to predict")
    con2.pack()
    con3=tk.Label(win,text="Record-to do real time recording and predict!Get started!")
    con3.pack()

#Change model window
def chg_model():
    win=tk.Toplevel(relief=tk.SUNKEN,borderwidth=2,height=200,width=300)
    win.title("Change model")
    lbl_model = tk.Label(master=win, text="Choose model:")
    ent_model = tk.Entry(master=win, width=50)
    lbl_model.grid(row=0, column=0, sticky="nsew")
    ent_model.grid(row=0, column=1)
    btn_submit = tk.Button(master=win, text="Upload",command=(lambda: upload(ent_upload)))
    btn_submit.grid(row=1,column=1, padx=10, ipadx=10)

#to direct to wiki page
def link(lbl):
    base="https://en.wikipedia.org/wiki/"
    link=base+wiki_path[lbl]
    webbrowser.open_new(link)
        
    
if __name__=="main":

    # Create a new window 
    window = tk.Tk()
    window.title("Bird Call Prediction")
    window.iconbitmap('icon.ico')
    
    #MENUS
    menu = tk.Menu(window)
    window.config(menu= menu)
    
    subm1 =tk.Menu(menu)
    menu.add_cascade(label="File",menu=subm1)
    subm1.add_command(label="Exit",command=window.destroy)
    
    subm2 = tk.Menu(menu)
    menu.add_cascade(label="Tools",menu=subm2)
    subm2.add_command(label="Change model",command=chg_model)
    subm2.add_command(label="Callibrate")
    
    subm3 = tk.Menu(menu)
    menu.add_cascade(label="Options",menu=subm3)
    subm3.add_command(label="About",command=abt)
    
    
    #Input form
    lbl_upload = tk.Label(master=window, text="Upload:") #upload option
    ent_upload = tk.Entry(master=window, width=50)
    lbl_upload.grid(row=0, column=0, sticky="nsew",pady = 2)
    ent_upload.grid(row=0, column=1,pady = 2)
    
    
    lbl_option=tk.Label(master=window, text="Or",width=50)
    lbl_option.grid(row=1,column=1,sticky="nsew")
    
    
    lbl_record = tk.Label(master=window, text="Seconds:") #record option
    ent_record = tk.Entry(master=window, width=50)
    lbl_record.grid(row=2, column=0, sticky="nsew",pady = 2)
    ent_record.grid(row=2, column=1,stick="nsew",pady = 2)
    
    
    #Images
    img= ImageTk.PhotoImage(Image.open("images.jpg"),master=window)
    canvas =tk.Label(master=window, width = 200, height = 200,image=img)        
    canvas.image=img  
    canvas.grid(row=0,column=2,columnspan=3,rowspan=2)
    
    #Buttons
    btn_submit = tk.Button(master=window, text="Upload",command=(lambda: upload(ent_upload)))
    btn_submit.grid(row=3,column=2,sticky="nsew")
    
    btn_clear = tk.Button(master=window, text="Record",command=(lambda: record(ent_record)))
    btn_clear.grid(row=3,column=3,sticky="nsew")
    
    window.mainloop()