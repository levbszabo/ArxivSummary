from rouge_score import rouge_scorer
import os
import glob
import gzip
import json
import random
import collections
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import struct
import six
import numbers
import re
from itertools import chain
import pathlib

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<S>'
SENTENCE_END = '</S>'

# Valid rouge types that can be computed are:
#       rougen (e.g. rouge1, rouge2): n-gram based scoring.
#       rougeL: Longest common subsequence based scoring.
# scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# scores = scorer.score('The quick brown fox jumps over the lazy dog', #targets
#                       'The quick brown dog jumps.')                    # decodes/prediction
# print(scores['rougeL'].precision)
def get_data(path,n=5):
    file = open(path,"r")                  
    lines = file.readlines()
    lines_subsets = lines[:n]
    article_dicts = []
    for f in lines_subsets:
        article_dicts.append(json.loads(f))
    file.close()
    return article_dicts

    
def match_sentences(abstract, sections, section_names):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    abstract_sections = [[] for i in range(len(section_names))] 
    for sent in abstract:
        # print(sent)
        max_p = -10
        max_sec = 0
        for i, sec in enumerate(section_names):
            for secs in sections[i]:
                scores = scorer.score(sent, secs)                    # decodes/prediction
                R = scores['rougeL'].precision
                if R > max_p:
                    max_p = R
                    max_sec = i

        abstract_sections[max_sec].append(sent)

    return abstract_sections

def convert_list_str(sections):
    for section in sections:    
        res = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in section])
        print(res)

if __name__ == '__main__':
    article_dicts = get_data("../data/val.txt") 
    print("Keys are ", article_dicts[0].keys()) #dict_keys(['article_id', 'article_text', 'abstract_text', 'labels', 'section_names', 'sections'])
    print("Section names are", article_dicts[0]['section_names'])
    print("The number of sections are", len(article_dicts[0]['sections']))
    # print("The first section is ", article_dicts[0]['sections'][0])

    final_list = []
    for article in article_dicts:
        article_dict = {}
        article_dict['abstract_text'] = article['abstract_text']
        article_dict['section_names'] = []
        article_dict['sections'] = []
        for i, sec_name in enumerate(article['section_names']):
            # print(i, sec_name)
            if sec_name in ['introduction', 'case', 'method', 'methods', 'techniques', 'methodology', 'conclusion', 'conclusions', 'concluding', 'discussion', 'limitations', 'summary']:
                article_dict['section_names'].append(sec_name)
                article_dict['sections'].append(article['sections'][i])
        final_list.append(article_dict)

    print('The final list is ', len(final_list))
    print('The keys in the final dict is ', final_list[0].keys())
    print(final_list[0]['section_names'])
    print(len(final_list[0]['sections']))

    for article in final_list:
        print('The keys are', article['section_names'])

    
    # print(len(final_list[0]['abstract_text'])) # 5
    # print(final_list[0]['abstract_text'][0])

    # classify each line of abstract into sections [{section:_, abstract:_}] based on ROUGE 
    names = []
    abstract_split = []

    abstracted_summaries = []
    for article in final_list:
        abstract = article['abstract_text']
        # print(abstract)
        str1 = ""
        abstract = str1.join(abstract).replace("<S>","").replace("</S>","").strip()
        abstract = abstract.strip().split(".")[:-1]
        section_names = article['section_names']
        sections = article['sections']
        # print('Sections are ', sections[0])
        abstract_sections = match_sentences(abstract, sections, section_names)
        print(len(abstract_sections))
        # print("Abstract section is ", abstract_sections)
        # abstracted_summaries.append(abstract_sections)
        # join the summaries and sections separated by <S>, </S>
        for i in range(len(sections)):
            if(abstract_sections[i] != []):
                abstract_str = convert_list_str(abstract_sections[i])
                section_str = convert_list_str(sections[i])

        

    
    # convert to binary etc





    