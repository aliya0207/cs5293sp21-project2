import numpy as np
import fnmatch
import re
from sklearn.linear_model import SGDClassifier


from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import glob
import io
import os
import pdb
import sys

import re
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

import random
from sklearn.feature_extraction import DictVectorizer
from sklearn import ensemble
import regex

def Read_files(text_files):
    # print(text_files)
    data = []
    filenames =[]
    path = os.getcwd()+"/"+text_files+"/*"
    for filename in glob.glob(path):
        #print(filename)
        filenames.append(filename.split('/')[-1])
        with open(filename, "r", encoding='utf-8') as f:
            data1 = f.read()
            data.append(data1)
    #print(filenames)
    return data, filenames

def get_redacted_entity(data):
    person_list=[]
    #person_list1=[]
    for sent in sent_tokenize(data):
        from nltk import word_tokenize
        x=word_tokenize(sent)
        for chunk in ne_chunk(pos_tag(x)):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':       
                a=""
                for c in chunk.leaves():
                    a=a+c[0]
                    a=a+" "
                person_list.append(a[:-1])
    count=len(person_list)            
    personlist1=set(person_list)
    person_list1=list(personlist1)
    #print(persons)
    #person_list1=sorted(person_list1, reverse= True)
    #print(person_list1)
    return person_list1

def training_features(text, person_name_list):
    features = []
    cc = len(text)
    wc = len(text.split())
    sc = len(sent_tokenize(text))
    cs = 0
    
    for i in text:
        if i == " ":
            cs+=1
    
    for i in range(0, len(person_name_list)):
        dict = {}
        dict['total_sentences'] = sc
        dict['total_words'] = wc
        dict['total_characters'] = cc
        dict['total_spaces'] = cs
        dict['length_of_name'] = len(person_name_list[i])
        dict['names_count'] = len(person_name_list)
        
        features.append(dict)

    return features

def testing_features(text, redacted_names_in_block):
    features = []
    cc = len(text)
    wc = len(text.split())
    sc = len(sent_tokenize(text))
    cs = 0
    
    for i in text:
        if i == " ":
            cs+=1
            
    for i in range(0, len(redacted_names_in_block)):
        dict = {}
        dict['total_sentences'] = sc
        dict['total_words'] = wc
        dict['total_characters'] = cc
        dict['total_spaces'] = cs
        dict['length_of_name'] = len(redacted_names_in_block[i])
        dict['names_count'] = len(redacted_names_in_block)
        
        features.append(dict)

    return features


def Redact(replace,data):
    for j in range(0,len(replace)):
        if replace[j] in data:
            length = len(replace[j])
            data = re.sub(replace[j], length*'\u2588', data, 1)
    return data

def remove_duplicate_names(names_list):
    names_list_unique = (set(names_list))
    names_list_unique = list(names_list_unique)
    #unique = [i for n, i in enumerate(names_list) if i not in names_list[:n]]
    #unique_namelist=unique
    return names_list_unique 
  
def Save_to_output_redacted(redact_result, folder, file_name):
    folder = os.getcwd() + '/'+folder+"/"
    new_file = file_name.replace(".txt", ".redacted.txt")
    isFolder = os.path.isdir(folder) 
    if isFolder== False:
        os.makedirs(os.path.dirname(folder))
    with open( os.path.join(folder, new_file), "w+", encoding="utf-8") as f:
        f.write(redact_result)
        f.close()
        
def Save_to_output_predicted(redact_result, folder, file_name, data_list, redacted_names):
    folder = os.getcwd() + '/'+folder+"/"
    isFolder = os.path.isdir(folder)
    print(folder)
    if isFolder== False:
        os.makedirs(os.path.dirname(folder))
    result = Get_predicted_output(redact_result, data_list, redacted_names)
    with open( os.path.join(folder, file_name), "w+", encoding="utf-8") as f:
        f.write(result)
        f.close()
        
def Get_predicted_output(redact_result, data_list, redacted_names):
    result =redact_result
    for i in range(0, len(data_list)):
        names = ""
        for j in data_list[i]:
            names += j
            names += ","
        names = names[:-1]
        result +="\n {} top 3 predicted names are {}".format(redacted_names[i], names) 
    return result
  

def Read_files2(text_files):
    # print(text_files)
    data = []
    filenames =[]
    for filename in glob.glob(os.getcwd()):
        print(filename)
        print(filename)
        filenames.append(filename.split('/')[-1])
        print(filenames)
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            data1 = f.read()
            data.append(data1)
    return data, filenames
  
  
def get_predictions(probability, redactions):
    total_predicted_words = []
    for x in range(0, len(redactions)):
        prob = probability[x]
        top_3_idx = np.argsort(prob)[-3:]
        predicted_words = []
        for i in range(0,3):
            index_range = top_3_idx[i]
            predicted_word = remove_duplicate_names[index_range]
            predicted_words.append(predicted_word)
        total_predicted_words.append(predicted_words)
    return (total_predicted_words)

if __name__=='__main__':
#train the model
    input_path = "Data"
    output_path_redacted = "redacted"
    output_path_prediction = "predicted"
    train_data, file_names = Read_files(input_path)
    print(file_names)
    replace_result_list = []
    names_list = []
    redacted_data_list=[]
    redacted_data=[]
    full_list_training_features = []
    full_list_names = []
    redacted_result = []
    for itr in range(0, len(train_data)):
    
        replace_result = get_redacted_entity(train_data[itr])
        #print(person_list_result)
        #replace_result = Fields_to_redact(person_list_result)
    
        redact_result = Redact(replace_result,train_data[itr])
        Save_to_output_redacted(redact_result, output_path_redacted, file_names[itr])
        redacted_data_list.append(redact_result)
        
        list_names_dict_features = training_features(train_data[itr], replace_result)
        full_list_training_features.extend(list_names_dict_features)
    
        full_list_names.extend(replace_result)
        
  

    v = DictVectorizer()
    X = v.fit_transform(full_list_training_features).toarray()
    full_list_names = np.array(full_list_names)
    model = svm.SVC(probability=True)
    #model = SGDClassifier()
    model.fit(X, full_list_names)
    names_unique = Get_Unique_Names(full_list_names)

   
    redacted_data, file_names = Read_files(output_path_redacted)
        
    for i in range(0, 12):
       # print(redacted_data[i])    
        redacted_names = re.findall(r'(\u2588+)', redacted_data[i])
        test_features = testing_features(redacted_data[i], redacted_names)
        if len(test_features) > 0:
            X_test = v.fit_transform(test_features).toarray()
            probability = model.predict_proba(X_test)
            total_predicted_words = get_predictions(probability, redacted_names)
            Save_to_output_predicted(redacted_data[i], output_path_prediction, file_names[i], total_predicted_words, redacted_names)
       
