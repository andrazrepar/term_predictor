from statistics import stdev
from conllu import parse_incr
import os
from nltk.tokenize import wordpunct_tokenize
from nltk import ngrams
import csv
import pandas as pd
from collections import Counter, OrderedDict
import math
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn import pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import svm
import pickle as pkl
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_indexer import indexer
from allennlp.commands.elmo import ElmoEmbedder
import time


def create_ngrams(lst, ngramLength):
    if len(lst) < ngramLength: # check if sentence is shorter than max_n gram
        max_n = len(lst)
    all_ngrams = []
    for i in range(1, ngramLength + 1):
        grams = ngrams(lst, i)
        for gram in grams:
            all_ngrams.append(gram)
    return all_ngrams

def filter_ngrams_after_analysis(lst):
    pattern = [word['upostag'] for word in lst]
    lemma = [word['lemma'].lower() for word in lst]
    if len(pattern) > 1 and pattern[-1] != 'NOUN': # patterns longer than 1 have to end with a noun to be terms
        return False
    elif pattern[0] not in ['ADJ', 'NOUN', 'ADV']: # patterns not starting with any of these are not terms
    #elif pattern[0] not in ['ADJ', 'NOUN']: # patterns not starting with any of these are not terms
        return False
    elif any(el in pattern for el in ['VERB', 'AUX', 'PART', 'CCONJ', 'PUNCT', 'SYM', 'X', 'DET']): # if pattern contains any of these, it is not a term
        return False
    elif len(pattern) == 1 and pattern[0] != 'NOUN': #only nouns can be single word terms
        return False
    elif len(' '.join(lemma)) < 4: # remove terms shorter than 4 characters
        return False
    elif any(',' in word for word in lemma): # to remove patterns such as this "10,70"
        return False
    elif any('_' in word for word in lemma): # to remove patterns such as this "_ osnsu1 _" which results from lemmatizing this: r_OsV1_LD
        return False
    else:
        return True

def prepare(conllu_texts_dir, ngramLength, min_freq):

    # load conllu texts
    domain_counts = {}
    candidates = {}
    sentences = []
    wordcount = 0

    for file in os.listdir(conllu_texts_dir):
            if file.endswith('.conllu'):
                with open(os.path.join(conllu_texts_dir,file), 'r') as f:
                    for sent in parse_incr(f):

                        # prepare sentences for index
                        sentences.append({
                            "content": ' '.join([token['lemma'] for token in sent]),
                            "raw": ' '.join([token['form'] for token in sent])
                        })

                        # count domain words
                        for token in sent:
                            wordcount += 1
                            if token['lemma'].lower() not in domain_counts:
                                domain_counts[token['lemma'].lower()] = 1
                            else:
                                domain_counts[token['lemma'].lower()] += 1

                        # count candidates
                        ngrams = create_ngrams(sent, ngramLength)
                        for ngram in ngrams:
                            if filter_ngrams_after_analysis(ngram):
                                lemma_ngram = ' '.join([token['lemma'].lower() for token in ngram])
                                if lemma_ngram not in candidates:


                                    candidates[lemma_ngram] = [ngram,]
                                else:
                                    candidates[lemma_ngram].append(ngram)

    idx = indexer.create_index('index_folder', 'rsdo_idx', schema=indexer.lemmatization_schema)
    indexer.add_sentences_to_index(sentences, idx)

    # remove candidates under frequency threshold
    filtered_candidates = {}
    words = []
    for candidate in candidates:
        if len(candidates[candidate]) < min_freq:
            continue
        else:
            filtered_candidates[candidate] = candidates[candidate]

            # get words for embeddings calculations
            for wrd in candidate.split():
                if wrd.lower() not in words:
                    words.append(wrd)

    return candidates, domain_counts, words, idx, wordcount, len(sentences)

def calculate_domain_embeddings(elmo, words, idx, sample_size):
    def embed_batch(sentences):
        # get tokens from conllu
        tokens = [[token for token in sentence[0].split()] for sentence in sentences]
        lemmas = [[token for token in sentence[1].split()] for sentence in sentences]
        wc = 0
        for s in tokens:
            wc = wc + len(s)
        print(wc)
        # calculate embeddings for tokens
        vectors = elmo.embed_batch(tokens)
        vectors = np.array([v[1] for v in vectors], dtype=object)
        return tokens, lemmas, vectors

    vocab = {}
    all_embs_dict = {}
    counter = 0
    
    for word in words:
        counter += 1
        start_time_index = time.time()
        sentences= indexer.query_sample(word, idx, sample_size=sample_size)
        #print(counter, "words index time, --- %s seconds ---" % (round((time.time() - start_time_index), 3)))
        start_time_elmo = time.time()
        tokens, lemmas, embeddings = embed_batch(sentences) # produce embeddings for sentences
        print(counter, "words embed time, --- %s seconds ---" % (round((time.time() - start_time_elmo), 3)))
        
        for i in range(len(tokens)): # for each line in batch
            for j in range(len(tokens[i])): # for each word/token in line
                if lemmas[i][j].lower() == word: # if lemma equals word
                    new_vec = embeddings[i][j]
                    if word in vocab:          # calculate running average
                        n = vocab[word][0]
                        curr_avg = vocab[word][1]
                        vocab[word][0] = n+1
                        vocab[word][1] = curr_avg + (new_vec - curr_avg)/(n+1)
                    else: # or just enter current value if not encountered yet
                        vocab[word] = [1, new_vec]

                    if word in all_embs_dict: # dict for stdev calculation
                        all_embs_dict[word].append(new_vec)
                    else:
                        all_embs_dict[word] = [new_vec]
        
            
        
        
        

    stdevs = {}
    for wrd in all_embs_dict:
        all_embs = np.array(all_embs_dict[wrd])
        
        emb_std = np.std(all_embs, axis=0) 
        stdevs[word] = np.mean(emb_std)

    return vocab, stdevs

def get_general_frequencies(file):
    counts = {}
    with open(file, 'r') as f:
        r = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
        total = int(next(r)[1])
        for line in r:
            counts[line[0]] = int(line[1])
    return counts, total

def get_term_element(form, element):
    result = []
    for word in form:
        result.append(word[element])
    return ' '.join(result)

def get_pattern_features(terms):
    termCopy = terms.copy()
    patternFeatures = {}
    udTags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    for term in terms:
        upos = [get_term_element(form, 'upostag') for form in terms[term]['termdata']]
        termCopy[term]['noUniqueUpos'] = len(list(set(upos))) # count number of unique upos 
        if len(list(set(upos))) == 1: # if only one upos, take first one
            upos = upos[0].split()
        else: # if more than 1, take the most common one
            upos = Counter(upos).most_common(1)[0][0].split()
        termCopy[term]['termLength'] = len(upos) # term length, based on number of ud tags
        for udTag in udTags:
            termCopy[term]['start_' + udTag] = 1 if udTag == upos[0] else 0 # if first word tag equals udTag, assign 1, else assign 0
            termCopy[term]['anywhere_' + udTag] = 1 if udTag in upos else 0 # if udTag in upos, assign 1, else assign 0
            termCopy[term]['end_' + udTag] = 1 if udTag == upos[-1] else 0 # if last word tag equals udTag, assign 1, else assign 0
            termCopy[term]['countOf_' + udTag] = upos.count(udTag)

    return termCopy


def calculate_termhood(terms, general_counts, general_total, domain_counts):
    termsCopy = terms.copy()
    # calculate total number of words in the domain corpus
    domain_total = 0
    for lemma in domain_counts:
        domain_total += domain_counts[lemma]

    frequencyFeatures = {}
    for term in terms:
        term_freq = len(terms[term]['termdata'])
        termsCopy[term]['term_freq'] = term_freq # add term frequency to term features
        #term_len = len(term.split())
        term_len = 0
        freq_square = math.pow(term_freq,2)
        #freq_square = term_freq
        sum_words = 0
        sum_words_general = 0
        sum_words_domain = 0
        noOfProps = 0
        for wrd in wordpunct_tokenize(term):
            if wrd in prepositions:
                noOfProps += 1
            elif wrd == '-':
                continue
            else:
                term_len = term_len + 1
                gen_freq = general_counts.get(wrd, 0)
                dom_freq = domain_counts.get(wrd, 0)
                gen_rel_freq = gen_freq/float(general_total) if gen_freq > 0 else 1/float(general_total)
                dom_rel_freq = dom_freq/float(domain_total)
                sum_words_general += gen_rel_freq
                sum_words_domain += dom_rel_freq
                rel_freq_ratio = dom_rel_freq/gen_rel_freq
                try:
                    log_word = math.log(rel_freq_ratio)
                except:
                    log_word = 0
                sum_words = sum_words + log_word
        termsCopy[term]['termGenFreq'] = sum_words_general
        termsCopy[term]['termDomFreq'] = sum_words_domain
        termhood = freq_square * sum_words #brez deljenja z dolžino termina, preferiramo mwu termine
        frequencyFeatures[term] = termhood
        
    return termsCopy

def transform_dataset(terms):
    feats = OrderedDict()
    elmo = OrderedDict()
    elmo2 = []
    for term in terms:
        singleTermFeats = {}
        singleTermFeats['term'] = term
        for feat in terms[term]:
            if feat == 'termdata':
                continue
            elif feat == 'elmo':
                elmo[term] = terms[term][feat][0]
                elmo2.append(terms[term][feat][0])
            else:
                singleTermFeats[feat] = terms[term][feat]
        feats[term] = singleTermFeats

    df = pd.DataFrame.from_dict(feats, orient='index')
    if elmoFeatures:
        elmoVectors = pd.DataFrame.from_dict(elmo, orient='index', columns=['ELMO' + str(i) for i in range(1, 2049)])
        df = df.join(pd.DataFrame(elmoVectors))
    return df

def compare_embeddings(terms, domain_vocab, general_vocab, gen_avg_all, prepositions, domain_term):
    termsCopy = terms.copy()
    
    
    lg = open('log.txt', 'w')
    for term in terms:
        words = term.split()
        tlen = 0
        domain_avg = np.zeros((1024,), dtype=np.float32).reshape(1, -1)
        general_avg = np.zeros((1024,), dtype=np.float32).reshape(1, -1)
        for word in words:
            if word in prepositions: # check if word in prepositions and omit
                continue
            else:
                tlen = tlen + 1
                try:
                    domain_vec = domain_vocab[word][1]
                except:
                    domain_vec = np.zeros((1024,), dtype=np.float32).reshape(1, -1)
                    lg.write('d\t'+term[0]+'\t'+word+'\n')
                
                domain_avg = domain_avg + domain_vec
                try:
                    general_vec = general_vocab[word]
                except:
                    #pripišemo 0 ali povprečni vektor
                    #general_vec = np.zeros((1024,), dtype=np.float32).reshape(1, -1)
                    general_vec = gen_avg_all
                    lg.write('g\t'+term[0]+'\t'+word+'\n')
                general_avg = general_avg + general_vec
        if tlen == 0:
            tlen = 1
        domain_avg = domain_avg/tlen
        general_avg = general_avg/tlen
        termsCopy[term]['elmo'] = np.concatenate((general_avg, domain_avg), axis=1)
        sim = cosine_similarity(domain_avg, general_avg)
        termsCopy[term]['elmoSim'] = sim[0][0]
        
        # compare to domain term
        domain_term_vector = domain_vocab[domain_term][1].reshape(1, -1)
        domain_sim = cosine_similarity(domain_avg, domain_term_vector)
        termsCopy[term]['elmoSim'] = domain_sim[0][0]

    return termsCopy



def get_general_vocab(general_corpus_embeddings):
    gen_avg_all = np.zeros((1024,), dtype=np.float32).reshape(1, -1)
    general_len = 0
    general_vocab = {}
    with open(general_corpus_embeddings, 'r') as gf:
        next(gf)
        next(gf) # preveri
        for line in gf:
            general_len = general_len + 1
            line = line.strip().split()
            wrd = line[0]
            vec = np.array( line[1:] ).astype('float32')
            vec = vec.reshape(1, -1)
            gen_avg_all = gen_avg_all + vec
            general_vocab[wrd] = vec

    gen_avg_all = gen_avg_all / general_len
    return general_vocab, gen_avg_all

def compare_stdevs(terms, stdevs, prepositions):
    termsCopy = terms.copy()
    avg_stdev = 0
    for term in terms:
        words = term.split()
        tlen = 0
        for word in words:
            if word in prepositions: # check if word in prepositions and omit
                continue
            else:
                tlen = tlen + 1
                try:
                    wrd_stdev = stdevs[word]
                except:
                    wrd_stdev = 0
                    
                avg_stdev = avg_stdev + wrd_stdev
        if tlen == 0:
            tlen = 1
        avg_stdev = avg_stdev / tlen
        termsCopy[term]['elmoStDev'] = avg_stdev
    return termsCopy

def create_features(candidates, generalCounts, generalTotal, general_vocab, general_avg_all, domain_counts, domain_vocab, domain_stdevs, domain_term):
    
    dataset = {}
    i = 0
    for term in candidates:
            dataset[term] = {'termdata': candidates[term], 'label': 0, 'filtered_out': 0, "dataset": "predict"}
            i = i + 1
       
    

    dataset = get_pattern_features(dataset)
    dataset = calculate_termhood(dataset, generalCounts, generalTotal, domain_counts)
    dataset = compare_embeddings(dataset, domain_vocab, general_vocab, gen_avg_all, prepositions, domain_term)
    #dataset = compare_embeddings(datasets[domain], domain_vocab, general_vocab, gen_avg_all, prepositions, domain_terms[domain]) # if elmo only, use this line
    dataset = compare_stdevs(dataset, domain_stdevs, prepositions)
    df = transform_dataset(dataset)
    #print(df.shape)
    return df

class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        hd_searches = hd_searches.drop(['term'], axis=1)
        return hd_searches.values

def predict_terms(model_path, dataset):
    clf = pkl.load(open(model_path, 'rb'))
    #print('set shape:', dataset.shape)
    X = dataset.drop(['label', "dataset"], axis=1)
    y_pred = clf.predict(X)
    result = pd.concat([dataset[['term', 'label']].reset_index(), pd.DataFrame(y_pred, columns=['prediction'])], axis=1)
    result.to_csv(conllu_texts_dir + "/results.csv", encoding="utf8", index=False)
    result = result.loc[result['prediction'] == 1]
    result.to_csv(conllu_texts_dir + "/results-positive.csv", encoding="utf8", index=False)
    
start_time = time.time()

generalCorpusUnigrams = 'resources/finalUnigramsGigafida.csv'
vocab_general = 'resources/ccgigafida_averaged_lemmas-stanford.txt'
model_path = "resources/all-domains.p"
conllu_texts_dir = "corpora/conllus_davcni/_conllu"
#conllu_texts_dir = "corpora/small_test"

prepositions = ['brez', 'do', 'iz', 'z', 's', 'za', 'h', 'k', 'proti', 'kljub', 'čez', 'skozi', 'zoper', 'po', 'o', 'pri', 'po', 'z', 's', 'na', 'ob', 'v', 'med', 'nad', 'pod', 'pred', 'za']
min_freq = 15
sample_size = 10
ngramLength = 5
frequencyFeatures = True
elmoFeatures = True
domain_term = "davek"

options_file = "resources/options.json"
weight_file = "resources/gigafida_weights.hdf5"

print("loading elmo ...", end=" ")
elmo = ElmoEmbedder(options_file, weight_file, -1)
print("done --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))

print("preparing candidates ...", end=" ")
candidates, domain_counts, words, idx, wordcount, sentence_count = prepare(conllu_texts_dir, ngramLength, min_freq)
print("done --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))
print("Total words:", wordcount)
print("Total sentences:", sentence_count)

print("loading general frequencies ...", end=" ")
generalCounts, generalTotal = get_general_frequencies(generalCorpusUnigrams)
print("done --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))


print("Words to embed:", len(words))
print("calculating domain embeddings ...")
domain_vocab, domain_stdevs = calculate_domain_embeddings(elmo, words, idx, sample_size)
print("done --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))

print("loading general vocab ...", end=" ")
general_vocab, gen_avg_all = get_general_vocab(vocab_general)
print("done --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))


print("Candidate count:", len(candidates))
print("creating features ...", end=" ")
df = create_features(candidates, generalCounts, generalTotal, general_vocab, gen_avg_all, domain_counts, domain_vocab, domain_stdevs, domain_term)
print("done --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))

print("loading elmo ...", end=" ")
predict_terms(model_path, df)
print("done --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))

print("Total time --- %s minutes ---" % (round((time.time() - start_time)/60, 3)))