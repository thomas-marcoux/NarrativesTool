from nltk import tokenize
import re
import json
import os
import pandas as pd
import sys
from tqdm import tqdm
import collections, functools, operator
from builtins import dict
from functions import pos_tag_narratives, run_comprehensive, entity_narratives, get_entities

stop_words = []
with open("stopwords.txt", "r", encoding="utf-8") as f:
    for line in f:
        stop_words.append(str(line.strip()))

def process_narratives(corpus):
    narratives = {}
    narratives = {}
    for text in tqdm(corpus, total=len(corpus), desc="Narratives"):
        item_narratives, item_entity_count, objectEntitiesList = process_posts(text)
        for key, value in item_narratives.items():
            if key in narratives:
                narratives[key].append(value)
                entity_count[key] += item_entity_count[key]
            else:
                narratives[key] = value
                entity_count[key] = item_entity_count[key]
    return narratives, narratives

def process_posts(text):
    countSentTotal = 0
    countSentFiltered = 0
    countSentFilteredTriplet = 0
    textSentString = ''
    ListSentences_Unique = []
    for tokens in tokenize.sent_tokenize(text):
        countSentTotal = countSentTotal + 1
        tokens = tokens.replace("’s", "s")
        """ Clean up activity"""
        tokens = re.sub(r"[-()\"#/@;:<>{}`'’‘“”+=–—_…~|!?]", " ", tokens)
        countSentFiltered = countSentFiltered +1
        if tokens not in ListSentences_Unique:
            ListSentences_Unique.append(tokens)
            textSentString += str(' ') + str(tokens)
            countSentFilteredTriplet = countSentFilteredTriplet + 1    
    ListSentences_Unique = []       
    tfidf_string = pos_tag_narratives(textSentString) #takes so much time
    result_scored = run_comprehensive(tfidf_string)
    sentences_scored = tokenize.sent_tokenize(result_scored)
    entity_count = {}
    objectEntitiesList = [x for x in get_entities(text) if x.lower() not in stop_words]
    data_narratives = entity_narratives(sentences_scored, objectEntitiesList, entity_count)
    return data_narratives, entity_count, objectEntitiesList

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'data')
    dir = os.path.join(data_dir, 'blogs-CE')
    for f in os.listdir(dir):
        if 'posts' in f:
            path = os.path.join(dir, f)
            df = pd.read_csv(path, usecols=['content'])
            corpus = list(df.iloc[:, 0])
            print(f'Processing {f}')
            narratives, narratives = process_narratives(corpus)
    print('Done')
