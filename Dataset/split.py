# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:10:10 2021

@author: Bharath Kumar
"""


import os
import json
import numpy as np
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
 

data_dir="data_set4"
json_path="file_paths_5_150.json"

signal={
        "paths":[],
        "labels":[],
        "class":[]
  }


def get_ratio(y1):
    arr=np.zeros(5)
    for v in y1:
        arr[v]+=1;
    return arr
    

def create_dict():
    classes=os.listdir(data_dir)
    label=0
    for class_dir in classes:
        cnt=0;
        print(f"Processing{class_dir}...label{label}..\n")
        class_path=os.path.join(data_dir,class_dir)
        for seg in os.listdir(class_path):
            print(f"seg is...{seg}..\n")
            signal["paths"].append(seg)
            signal["labels"].append(label)
            signal["class"].append(class_dir)
            cnt+=1
        print(f"no.of files in {class_dir} is {cnt}..")
        label+=1  
        
    X=np.array(signal["paths"])
    y=np.array(signal["labels"])
    y_=np.array(signal["class"])
    
    n=input("Save the filepaths and classes?(y/n)")
    if(n=='y'):
             with open(json_path, "w") as fp:
                  json.dump(signal,fp, indent=4)
        
    return X,y,y_,classes

def test_split(X,y_,test_size):
    """X,y_:Arrays with input samples(total) and corresponding classnames
    Creates a Test dir and returns X,y_ excluding these Test samples"""
    
    #train/test split
    ts = StratifiedShuffleSplit(test_size=test_size, random_state=0)
    train_index,test_index= next(ts.split(X,y_))
    test_path=os.path.join(data_dir,"Test")
    
    print(f"No.of Test files{len(test_index)}...")
    
    for ind in test_index:
        file_name=X[ind]
        class_name=y_[ind]
        class_path=os.path.join(test_path,class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        dst_path=os.path.join(test_path,class_name,file_name)
        src_path=os.path.join(data_dir,class_name,file_name)
        shutil.move(src_path,dst_path)
        print(f"{file_name}Moved to Test dir...")
    
    #delete test samples
    X=np.delete(X,test_index)
    y_=np.delete(y_,test_index)
    
    #return remaining X&y_
    return X,y_


def valid_split(X,y_,valid_size):
    #train/valid split
    vs = StratifiedShuffleSplit(test_size=valid_size, random_state=0)
    train_index,valid_index= next(vs.split(X,y_))
    validation_path=os.path.join(data_dir,"Valid")
    
    print(f"No.of Valid files..{len(valid_index)}..")
    for ind in valid_index:
        file_name=X[ind]
        class_name=y_[ind]
        class_path=os.path.join(validation_path,class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        dst_path=os.path.join(validation_path,class_name,file_name)
        src_path=os.path.join(data_dir,class_name,file_name)
        shutil.copy(src_path,dst_path)
        print(f"{file_name}Moved to Valid dir...")
        
    
def valid_test_move():    
         X,y,y_,classes=create_dict()
         n=input("1.Valid Split\n 2.Test Split..")
         print(n)
         if(n=='1'):
             valid_size=float(input("Enter Valid data size(bw 0 to 1)..."))
             valid_split(X,y_,valid_size)
         elif(n=='2'):
              test_size,valid_size=[float(x) for x in input("Enter test and valid data size(bw 0 to 1)...").split()]
              X,y_=test_split(X,y_,test_size)
              valid_split(X,y_,valid_size)
         else:
             print("Invalid choice")
           
        #creating Train dir
         train_path=os.path.join(data_dir,"Train")
         os.makedirs(train_path)
         for class_dir in classes:
            print(f"class{class_dir}..")
            dst_path=os.path.join(train_path)
            src_path=os.path.join(data_dir,class_dir)
            shutil.move(src_path,dst_path)
            print(f"{class_dir}Moved to train..")
        
        
             
if __name__=="__main__":
    
    valid_test_move()
    #valid_test_move(X,y_,test_ind,valid_ind)
    
  
     
    """for train_index,test_index in sss.split(X,y):
         train_arr=get_ratio(y[train_index])
         test_arr=get_ratio(y[test_index])
         print("train...")
         print(train_arr)
         print()
         print("test..")
         print(test_arr)
         print()""" 
    
  