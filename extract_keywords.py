"""
Extract keywords from the tagged data.
For now: Only step 3b = Fragestellung nennen
Trying out a few different methods:
- TF-IDF

Sources
https://monkeylearn.com/keyword-extraction/
"""
import os
from preprocessing import Sentence
from TFIDF import create_tf_matrix, create_idf_matrix, create_tf_idf_matrix
from rake_nltk import Rake
from summa import keywords
import yake
from collections import Counter

import math
from tqdm.notebook import tqdm
from re import sub
from itertools import islice
#from sklearn.feature_extraction.text import TFidfVectorizer


def readfile(file):
    """Input: text file; Output: Tuple (step_name, list of sentences tagged with that step)"""
    file_name = os.path.basename(file)
    step_name = file_name[2] + file_name[9]  # TODO check if naming pattern applies to other files as well
    with open(os.path.join("data", file), 'r') as infile:
        lines = infile.readlines()
        lines = [line.rstrip('\n') for line in lines if line != "\n"]
        return step_name, lines


def preprocess(lines):
    """Takes list of sentences and returns list of preprocessed sentences."""
    preprocessed = [Sentence(sent).preprocessed for sent in lines]
    return preprocessed

def get_most_freq_words(sentences):
    all_words = []
    for s in sentences:
        words = s.split()
        for word in words:
            all_words.append(word.lower())
    wordfreq = [all_words.count(w) for w in all_words]
    pairs = list(zip(all_words, wordfreq))
    pairs = removeDuplicates(pairs)
    pairs.sort(key=lambda a: a[1], reverse=True)
    return pairs

def removeDuplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]

def keywords_TFIDF(sentences, threshold):
    """Keyword extraction using TF-IDF vectors (inverse score).
    Code source: https://medium.com/@ashins1997/text-summarization-f2542bc6a167
    Adapted by Laura Stahlhut"""
    # calculate idf score for all sentences in 1 file
    tf = create_tf_matrix(sentences)  # tf matrix
    idf = create_idf_matrix(sentences)  # idf matrix
    tf_idf = create_tf_idf_matrix(tf, idf)  # tf-idf matrix

    #we need to take the inverse somehow
    #for a quick approximation, i'm just setting a lower threshold of 0.04 (fragestellung is usually around 0.02
    candidates = []
    for k, v in tf_idf.items():
        for k2, v2 in v.items():
            #candidates are all words with a tf-idf score which is lower than a threshold
            if v2 <= threshold:
                candidates.append(k2)
    candidates_cnt = Counter(candidates)  # count how often each candidate occurs
    return candidates_cnt


def keywords_TextRank(sentences):
    text = ' '.join(sentences) # Treating the sentences like one big text
    TR_keywords = keywords.keywords(text, scores=True)
    return TR_keywords


def keywords_RAKE(sentences):
    """Rapid Automatic Keyword Extraction"""
    rake = Rake()
    rake.extract_keywords_from_sentences(sentences)
    keywords = rake.get_ranked_phrases()
    kw = ([keyword for keyword in keywords if
           len(keyword.split()) > 1 and len(keyword.split()) < 8])  # at least two words long
    return kw


def keywords_yake(sentences):
    text = ' '.join(sentences)  # Treating the sentences like one big text
    kw_extractor = yake.KeywordExtractor(top=10)
    keywords = kw_extractor.extract_keywords(text)

    return keywords

def keywords_CRG():
    # TODO
    pass

def main():
    # get a list of preprocessed sentences tagged with step 3b
    # (in a list so that it works when we have > 1 file later on)
    filenames = os.listdir("data")
    all_data = [readfile(f) for f in filenames]  # [(step, list of sentences), ...] -> potentially for all steps
    all_data_preprocessed = [(step[0], preprocess(step[1])) for step in all_data]  #[(step, preprocessed sentences),...]


    # Keyword extraction
    for step in all_data_preprocessed:
        if step[0] == '3b':
            sentences = step[1]  # list of preprocessed sentences tagged with step 3b

            # keyword extraction based on TF-IDF scores
            print("\nKeywords based on TF-IDF: ")
            print(keywords_TFIDF(sentences, 0.03))

            # keyword extraction with RAKE
            print("\nKeywords produced with RAKE:")
            print(keywords_RAKE(sentences)) # funktioniert nicht so gut

            # keyword extraction with TextRank
            print("\nKeywords produced with TextRank:")
            print(keywords_TextRank(sentences)[0:10]) # funktioniert ein bisschen besser

            # keyword extraction with YAKE!
            print("\nKeywords produced with YAKE:")
            for kw, v in keywords_yake(sentences):
                print("Keyphrase: ", kw, ": score", v)

            # most frequent words
            print("\n'Keywords' as in Most frequent words in this preprocessed corpus: ")
            print(get_most_freq_words(sentences)[0:20])



if __name__ == '__main__':
    main()