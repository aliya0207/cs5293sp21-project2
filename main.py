import numpy as np
import fnmatch
import re
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


def redact_data(replace,data):
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
    result = redact_result
    for i in range(0, len(data_list)):
        names = ""
        for j in data_list[i]:
            names += j
            names += ","
        names = names[:-1]
        result +="\n {} top 3 predicted names are {}".format(redacted_names[i], names) 
    return result

def get_predictions(probability, redactions, names_unique):
    total_predicted_words = []
    for x in range(0, len(redactions)):
        prob = probability[x]
        top_3_idx = np.argsort(prob)[-3:]
        predicted_words = []
        for i in range(0,3):
            index_range = top_3_idx[i]
            predicted_word = names_unique[index_range]
            predicted_words.append(predicted_word)
        total_predicted_words.append(predicted_words)
    return (total_predicted_words)

def train_models(full_list_training_features, full_list_names):
    dv = DictVectorizer()
    train = dv.fit_transform(full_list_training_features).toarray()
    full_list_names = np.array(full_list_names)
    model = svm.SVC(probability=True)
    model.fit(train, full_list_names)
    return dv, full_list_names, model
    
if __name__=='__main__':
#train the model
    input_path = "Data"
    output_path_redacted = "redacted"
    output_path_prediction = "predicted"
    #reading all the files at once and extracting their names 
    train_data, file_names = Read_files(input_path)
    replace_result_list = []
    names_list = []
    redacted_data_list=[]
    redacted_data=[]
    full_list_training_features = []
    full_list_names = []
    redacted_result = []
    #iterating over each files, redact them and get the redacted names
    for itr in range(0, len(train_data)):
        #getting redacted entities
        replace_result = get_redacted_entity(train_data[itr])
        #redact the files
        redact_result = redact_data(replace_result,train_data[itr])
        redacted_data_list.append(redact_result)
        #saving each redacted files back into the directory
        Save_to_output_redacted(redact_result, output_path_redacted, file_names[itr])
        #creating dictionary of names with its features from the file
        list_names_dict_features = training_features(train_data[itr], replace_result)
        #creating list of all the features 
        full_list_training_features.extend(list_names_dict_features)
        full_list_names.extend(replace_result)

    dv, full_list_names, model = train_models(full_list_training_features, full_list_names)
    names_unique = remove_duplicate_names(full_list_names) 
    redacted_data, file_names = Read_files(output_path_redacted)
    
    for i in range(0, 12):
       # print(redacted_data[i])    
        redacted_names = re.findall(r'(\u2588+)', redacted_data[i])
        test_features = testing_features(redacted_data[i], redacted_names)
        if len(test_features) > 0:
            X_test = dv.fit_transform(test_features).toarray()
            probability = model.predict_proba(X_test)
            total_predicted_words = get_predictions(probability, redacted_names,names_unique)
            Save_to_output_predicted(redacted_data[i], output_path_prediction, file_names[i], total_predicted_words, redacted_names)
       
