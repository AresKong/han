# encoding: utf-8
import os
import nltk
from collections import defaultdict
import json
import re

# Clean uttr / Remove non ascii
def clean_uttr(uttr):
    uttr = uttr.strip()
    uttr = uttr
    uttr = re.sub(r'[^\x00-\x7f]', r'', uttr)
    return uttr

def get_vocabulary(filepath):
    vocabulary = set()
    for filename in os.listdir(filepath):
        if filename.endswith(".txt"):
            with open(os.path.join(filepath, filename)) as file:
                ids = filename.rstrip(".txt")
                for line in file:
                    uttr = clean_uttr(line)
                    tokens = nltk.word_tokenize(uttr)
                    for w in tokens:
                        vocabulary.add(w)
    return vocabulary

def get_diag(filename):
    diag = {}
    with open(filename) as file:
        for line in file:
            l = line.split()
            diag[l[0].rstrip("c")] = l[1]
    return diag

def get_age(filename):
    age = {}
    with open(filename) as file:
        for line in file:
            l = line.split()
            age[l[0].rstrip("c")] = l[1]
    return age    

# Extract data from dbank directory
def get_json_data(word2index, diag, age, filepath):
    parsed_data = {"ids":[], "sentences":[], "tokens":[], "labels":[], "age":[]}
    balanced_class = 0
    classes = ["Control", "ProbableAD", "PossibleAD"]

    for filename in os.listdir(filepath):
        if filename.endswith(".txt"):
            with open(os.path.join(filepath, filename)) as file:
                ids = filename.rstrip(".txt")
                if ids not in diag:
                    continue

                diagnosis = diag[ids]
                if diagnosis not in classes:
                    continue

                if diagnosis == "Control":
                    label = 0
                else:
                    label = 1

                print("Parsing: " + ids)
                parsed_data["ids"].append(ids[:3])
                sentences = []
                numberized_sent = []

                for line in file:
                    uttr = clean_uttr(line)
                    tokens = nltk.word_tokenize(uttr)
                    numberized = [word2index[w] for w in tokens]
                    
                    uttr = " ".join(tokens)
                    sentences.append(uttr)
                    numberized_sent.append(numberized)

                parsed_data["sentences"].append(sentences)
                parsed_data["tokens"].append(numberized_sent)
                parsed_data["labels"].append(label)
                if ids not in age or age[ids] == 'NaN':
                    if diag[ids] == "Control":
                        age[ids] = 63.95
                    else:
                        age[ids] = 71.72
                parsed_data["age"].append(int(age[ids]))
    else:
        print("Filepath not found: " + filepath)
        print("Data may be empty")
    return parsed_data

def build_dbank_json():
    vocabulary = get_vocabulary('./data/dementiabank')
    print(len(vocabulary))
    word2index = {word:index+1 for index, word in enumerate(vocabulary)}
    diag = get_diag('./data/dementiabank_info/diagnosis.txt')
    age = get_age('./data/dementiabank_info/age_gender.txt')
    data = get_json_data(word2index, diag, age, './data/dementiabank')
    with open('dbank.json', 'w') as outfile:
        json.dump(data, outfile)

