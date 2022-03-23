import os
from whoosh import index
from whoosh.fields import Schema, TEXT, STORED
from whoosh.analysis import Filter, SimpleAnalyzer, PassFilter
from whoosh.qparser import QueryParser
from random import sample

from lemmagen3 import Lemmatizer

lem_sl = Lemmatizer('sl')


class LemmatizationFilter(Filter):
    is_morph = True

    def __init__(self, lang):
        self.lang = lang

    def __call__(self, tokens):
        for t in tokens:
            t.text = lem_sl.lemmatize(t.text)
            yield t


simple_schema = Schema(content=TEXT(analyzer=SimpleAnalyzer(), stored=True),
                       raw=STORED())
lemmatization_schema = Schema(content=TEXT(analyzer=SimpleAnalyzer() | LemmatizationFilter(lang='sl'), stored=True),
                              raw=STORED())


def read_or_create_index(idirname, iname, schema):
    if not os.path.exists(idirname):
        os.mkdir(idirname)
        ix = index.create_in(idirname, schema=schema, indexname=iname)
    try:
        ix = index.open_dir(idirname, indexname=iname)
    except:
        ix = index.create_in(idirname, schema=schema, indexname=iname)
    return ix


def create_index(idirname, iname, schema=lemmatization_schema):
    if not os.path.exists(idirname):
        os.mkdir(idirname)
    ix = index.create_in(idirname, schema=schema, indexname=iname)
    return ix


def add_sentences_to_index(sentences, index):
    with index.writer() as writer:
        for sentence in sentences:
            if isinstance(sentence, str):
                writer.add_document(content=sentence, raw=sentence)
            elif isinstance(sentence, dict):
                if 'content' in sentence and 'raw' in sentence:
                    writer.add_document(content=sentence['content'], raw=sentence['raw'])
                else:
                    raise ValueError('When unput sentences are of type dict they must have these two keys: "content" and "raw". content is indexed but not stored, raw is stored but not indexed')
            else:
                raise ValueError('Input sentences must be either strings or dicts with keys "content" and "raw"')


def query(query, index, query_field='content', limit=10):
    qp = QueryParser(query_field, schema=index.schema)
    q = qp.parse(query)
    with index.searcher() as s:
        results = s.search(q, limit=limit)
        sentences = [x['raw'] for x in results]
    return sentences


def query_sample(query, index, query_field='content', sample_size=10):
    qp = QueryParser(query_field, schema=index.schema)
    q = qp.parse(query)
    with index.searcher() as s:
        results = s.search(q, limit=None)
        sentences = [(x['raw'], x['content']) for x in results]
        filtered_sentences = [sent for sent in sentences if len(sent[1].split()) > 10 and len(sent[1].split()) < 15]
        #print(query, len(sentences), len(filtered_sentences))
        #print(sentences[0], filtered_sentences[0])
        
    
    if len(filtered_sentences) >= sample_size:
        print(query, "filtered", len(sentences), len(filtered_sentences))
        if sample_size <= len(filtered_sentences):
            return sample(filtered_sentences, sample_size)
        else:
            return filtered_sentences
    else:
        print(query, "not filtered", len(sentences), len(filtered_sentences))
        if sample_size <= len(sentences):
            return sample(sentences, sample_size)
        else:
            return sentences
