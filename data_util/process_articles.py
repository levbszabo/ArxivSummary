import re
import numpy as np
import json
from rouge_score import rouge_scorer
from argparse import ArgumentParser
from nltk.translate.bleu_score import corpus_bleu


#We specify the keywords for each section
section_keywords = {'introduction':['introduction', 'case'],
'literature': ['background', 'literature', 'related'],
'methods': ['method','methods','technique','techniques','methodology'],
'results': ['result', 'results', 'experimental', 'experiment','experiments'],
'conclusion': ['conclusion','conclusions', 'concluding', 'discussion', 'summary','limitations']}
keys = [k for k in section_keywords]


#We find the number of keyword matches for each current section, then we find 
#the new section name which can serve as a label
def match_section(sec_name):
    keys = [k for k in section_keywords]
    counts = [0 for k in keys]
    for i,k in enumerate(keys):   
        kw = section_keywords[k]
        kw_str = "|".join(kw)  
        regex_str = "({})".format(kw_str)
        matches = re.findall(regex_str, sec_name)
        counts[i] = sum([1 for i in matches])
    if sum(counts) == 0:
        return None
    else:
        return keys[np.argmax(counts)]  
#Using a rouge or bleu scorer we group all the abstract sentences
def match_abstract(abstract,sections,metric='rouge'):
    abstract_groupings = [[] for _ in keys]
    if metric == 'rouge':
        scorer = rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True) # why rougeLsum?
        for sent in abstract:
            sent_clean = re.sub(r"<S>|</S>|\n","",sent).strip()
            rouge_scores = []
            for i in range(len(sections)):
                if sections[i] == None:
                    rouge_scores.append(0)
                else:
                    sc = max([scorer.score(sections[i][j],sent)['rougeL'].precision for j in range(len(sections[i]))])
                    rouge_scores.append(sc)
            section_label = np.argmax(rouge_scores)
            abstract_groupings[section_label].append(sent)
    else:
        for sent in abstract:
            sent_clean = re.sub(r"<S>|</S>|\n","",sent).strip()
            sent_clean = sent_clean.split(" ")
            bleu_scores = []
            for i in range(len(sections)):
                if sections[i] == None:
                    bleu_scores.append(0)
                else:
                    sec_formatted = [[sections[i][j].strip().split(" ") for j in range(len(sections[i]))]]
                    bleu_score = corpus_bleu(sec_formatted,[sent_clean])
                    bleu_scores.append(bleu_score)
            section_label = np.argmax(bleu_scores)
            abstract_groupings[section_label].append(sent)
    print([len(v) for v in abstract_groupings])
    return abstract_groupings 
        
    
# for a given article d as dictionary we filter sections and mark abstracts
# output is a list of strings encoded as dictionaries, each of these is a "new" article
def generate_summaries(d,metric='rouge'):
    outputs = []
    section_names = d["section_names"]
    section_text = np.array(d["sections"])
    section_classifications = np.array([match_section(sec) for sec in section_names])
    new_section_text = []
    for group in keys:
        if group == "literature":
            new_section_text.append(None)
            continue
        locs = np.where(section_classifications==group)[0]
        text = section_text[locs]
        merged_text = " ".join([" ".join(v) for v in text])
        if len(merged_text) <= 1:
            new_section_text.append(None)
        else:
            merged_text = merged_text.split(".")
            new_section_text.append(merged_text)
    
    abstract = d["abstract_text"]
    abstract_groupings = match_abstract(abstract,new_section_text,metric)
    new_section_t = []
    for i in range(len(new_section_text)):
        if new_section_text[i] != None:
            val = " ".join(new_section_text[i])
            new_section_t.append(val)
        else:
            new_section_t.append(None)
    new_section_text = new_section_t
    # print(abstract_groupings)
    for i,a in enumerate(abstract_groupings):
        if len(a) != 0 and new_section_text[i] != None:
            # article_id = d["article_id"]
            # article_text = [new_section_text[i]]
            abstract_text = abstract_groupings[i]
            # labels = [abstract_groupings[i]]
            # section_names = [keys[i]]
            sections = [new_section_text[i]]
            # data = {'article_id':article_id,'article_text':article_text,'abstract_text':abstract_text[0],
            #                   'labels':labels[0],'section_names':section_names,'sections':sections}
            # print("data is ", data)
            data = {'abstract_text':abstract_text, 'sections':sections}
            outputs.append(json.dumps(data))
            # outputs.append(data)
    # print(outputs[1]['abstract_text'])
    # print('########################')
    # print(outputs[1]['sections'])
    return outputs

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('infile', help='path to the jsonlines data')
    ap.add_argument('outfile', help='path to the output file')
    ap.add_argument('--num',type = int,default = -1, help ='number of articles to process')
    ap.add_argument('--metric', type = str, default = 'rouge',help='type of metric used to split abstract')
    args = ap.parse_args()
    #We read in a text file and apply section filtering and abstract matching
    out = open(args.outfile, 'w+', encoding='utf-8') 
    out.close()
    if args.num == -1:
        num_articles = sum([1 for _ in open(args.infile)])
    else:
        num_articles = args.num
    with open(args.infile,"r") as f:
        for i in range(num_articles): 
            print("Article no:", i, " of ", num_articles)
            line = f.readline()
            if not line.strip():
                continue
            line = line.strip()
            data = json.loads(line)
            data_outputs = generate_summaries(data,args.metric)
            with open(args.outfile,"a",encoding='utf-8') as out:
                for data_out in data_outputs:
                    out.write(data_out+"\n")
                out.close()
        f.close()
        out.close()