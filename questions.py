__author__ = "Gabriela Karina Pauljus"
__license__ = "GPL"
__version__ = "1.0.1"
__status__ = "Prototype"

import nltk
import sys
import os
import pandas as pd
import string
import math
import time

FILE_MATCHES = 4
SENTENCE_MATCHES = 3


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    while True:

        # Prompt user for query
        query = set(tokenize(input(f"\nAsk me anything antibiotic resistance or enter a term of interest (e.g. Rhine): ")))

        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(f'\n{match}')

        follow_up = input(f"\nDo you want to ask another question? [Y/N]: ")
        while True:
            if (follow_up.lower() == 'y' ) or (follow_up.lower() == 'yes' ):
                break
            elif (follow_up.lower() == 'n' ) or (follow_up.lower() == 'no' ):
                print("Ok, see you soon!")
                sys.exit(1)
            else:
                follow_up = input(f'"{follow_up}" is not a valid option, dummie! Enter Y or N!!: ')



def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # initiate dict
    topics = {}

    # define path to directory
    path_to_dir = os.path.join(".", f"{directory}")

    # iterate over files in directory
    for file in os.listdir(path_to_dir):

        # get file path
        path_to_file = os.path.join(path_to_dir, file)

        # read file into "string"
        with open(path_to_file,"r", encoding="unicode_escape") as f:
            string = f.read()

        # save text for each file in topics-dict; key is the file-name EXCLUDING the .txt-extension
        topics[file[:-4]] = string

    # return dict
    return topics

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.
    """
    # extract tokens and save lowercase tokens in tokens list
    tokens = [word.lower() for word in nltk.word_tokenize(document)]

    # initiate filtered-list
    filtered = []

    # define stopwords and puntuation
    stopwords = nltk.corpus.stopwords.words("english")
    punct = [punct for punct in string.punctuation]

    # iterate over tokens and filter out stopwords and punctuation
    for word in tokens:
        if word in stopwords:
            continue
        elif word in punct:
            continue
        else:
            filtered.append(word)

    # # create distribution dictionary to check how often any given word of interst appears in document
    # distrib = {}
    # for word in filtered:
    #     if word in distrib.keys():
    #         distrib[word] +=1
    #     else:
    #         distrib[word] = 1

    # # check 15 most frequent words in document
    # x = 0
    # for w in sorted(distrib, key=distrib.get, reverse=True):
    #     if x < 100:
    #         x+=1
    #         print(w, distrib[w])

    # return filtered list
    return(filtered)



def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, returns a dictionary that maps words to their IDF values.
    """
    # number of dictionaries/files
    numDict = len(documents)

    # initiate presence-dict --> saves the number of documents that a specific word is present in
    presence = {}

    # initiate idfs-dict to save calculated idfs in
    idfs = {}

    # iterate over docs and unique words within each doc
    for doc in documents:
        for word in set(documents[doc]):
            # presence = {key: 1 if key not in presence.keys else key: presence[key]+=1 for key in set(documents[doc])}
            if word in presence.keys():
                presence[word] += 1
            else:
                presence[word] = 1

    for word in presence:
        idf = 1 + (math.log(numDict/(presence[word]))) #add one for smoothing
        idfs[word] = idf

    # return idf-dict
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # initiate tfidf-dict
    tfidfs = {}

    # iterate over files and words in query and calculate tfidf per file-queryword pair
    for file in files:
        tfidfs[file] = 0
        tokens_in_file = len(files[file])
        for word in query:
            if word in files[file]:
                frequency = files[file].count(word)+1 # '+1' for smoothing
            else:
                frequency = 1
            tf = frequency/tokens_in_file         # normalize frequency to account for different length of texts  
            if word in idfs.keys():
                idf = idfs[word]
            else:
                idf = 1
            tfidfs[file] += idf * tf       # sum tfidfs from different word together per file

    # list with sorted files (from most relevant to least depending on tfidf-values in dict)
    sorted_list = sorted(tfidfs, key=tfidfs.get, reverse=True)

    # create list with n top files
    topFiles = sorted_list[:n]

    # return topFiles
    return topFiles

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference is given to sentences that have a higher query term density.
    """
        # initiate tfidf-dict
    sentence_stats = {}

    # iterate over files and words in query and calculate tfidf per file-queryword pair
    for sentence in sentences:
        sentence_stats[sentence] = {}
        sentence_stats[sentence]['idf'] = 0
        sentence_stats[sentence]['word_count'] = 0
        senlength = len(sentences[sentence])
        for word in query:
            if word in sentences[sentence]:
                sentence_stats[sentence]['idf'] += idfs[word]
                sentence_stats[sentence]['word_count'] += 1
        sentence_stats[sentence]['QTD'] = float(sentence_stats[sentence]['word_count'] / senlength)

    # list with sorted sentences (from most relevant to least depending on (first) idf and (second) density)
    sorted_list = sorted(sentence_stats.keys(), key=lambda sentence: (sentence_stats[sentence]['idf'], sentence_stats[sentence]['QTD']), reverse=True)

    # create list with n top sentences
    topSens = sorted_list[:n]

    # return topSens
    return topSens

if __name__ == "__main__":
    main()
