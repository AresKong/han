# encoding: utf-8
import os
import nltk
from collections import defaultdict
import json
import xml.etree.cElementTree as ET
import re
import pandas as pd


DEMENTIA_BLOGS = ["creatingmemories", "living-with-alzhiemers", "parkblog-silverfox"]

def isValid(inputString):
    # Line should not contain numbers
    if(any(char.isdigit() for char in inputString)):
        return False
    # Line should not be empty
    elif not inputString.strip():
        return False
    # Line should contain characters (not only consist of punctuation)
    elif not bool(re.search('[a-zA-Z]', inputString)):
        return False
    else:
        return True

# Clean uttr / Remove non ascii
def clean_uttr(uttr):
    uttr = uttr.strip()
    uttr = uttr
    uttr = re.sub(r'[^\x00-\x7f]', r'', uttr)
    return uttr

def get_vocabulary(filepath):
    vocabulary = set()
    tree = ET.parse(filepath)
    root = tree.getroot()
    cutoff_date = pd.datetime(2017, 4, 4)
    for blog in root:
        for post in blog:
            quality = post.attrib['quality']
            if quality != "good":
                continue
            date = pd.to_datetime(post.attrib['date'])
            if date > cutoff_date:
                continue

            for sentence in post:
                sentence = sentence.text
                if isValid(sentence):
                    uttr = clean_uttr(sentence)
                    tokens = nltk.word_tokenize(uttr)
                    words = [w.lower() for w in tokens]
                    for w in words:
                        if w not in vocabulary:
                            vocabulary.add(w)

    return vocabulary

def get_json_data(word2index, filepath):
    parsed_data = {"blogs":[], "sentences":[], "tokens":[], "labels":[], "ids":[]}
    tree = ET.parse(filepath)
    root = tree.getroot()
    cutoff_date = pd.datetime(2017, 4, 4) 
    for blog in root:
        name = blog.attrib['name'].replace('http://', '').replace('https://', '').replace('.blogspot.ca', '')
        for post in blog:
            quality = post.attrib['quality']
            if quality != "good":
                continue
            date = pd.to_datetime(post.attrib['date'])
            if date > cutoff_date:
                continue
            post_id = post.attrib['id']
            print("processing", post_id)
            parsed_data["blogs"].append(name)
            parsed_data["ids"].append(name+"_"+post_id)
            if name in DEMENTIA_BLOGS:
                label = 1
            else:
                label = 0
            parsed_data["labels"].append(label)
            sentences = []
            numberized_sent = []
            for sentence in post:
                sentence = sentence.text
                if not isValid(sentence):
                    continue
                uttr = clean_uttr(sentence)
                tokens = nltk.word_tokenize(uttr)
                words = [w.lower() for w in tokens]
                numberized = [word2index[w] for w in words]
                uttr = " ".join(words)
                sentences.append(uttr)
                numberized_sent.append(numberized)
            parsed_data["sentences"].append(sentences)
            parsed_data["tokens"].append(numberized_sent)
    return parsed_data

def build_blog_json():
    vocabulary = get_vocabulary('./data/blog_corpus.xml')
    # print(len(vocabulary)) len = 27413
    word2index = {word:index+1 for index, word in enumerate(vocabulary)}
    data = get_json_data(word2index, './data/blog_corpus.xml')
    with open('blog_data.json', 'w') as outfile:
        json.dump(data, outfile)
