import re
import numpy as np
import json
from rouge_score import rouge_scorer
from argparse import ArgumentParser

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

#Using a rouge scorer we group all the abstract sentences
def match_abstract(abstract,sections):
    scorer = rouge_scorer.RougeScorer(['rougeLsum'],use_stemmer=True) # why rougeLsum?
    abstract_groupings = [[] for _ in keys]
    sec_strings = []
    rouge_scores_list = []
    for sec in sections:
        if sec == None:
            sec_strings.append("")
        else:
            sec = [re.sub(r"\n","",v).strip() for v in sec]
            sec = "\n".join(sec)
            sec_strings.append(sec)
    for sent in abstract:
        sent_clean = re.sub(r"<S>|</S>|\n","",sent).strip()
        rouge_scores = [scorer.score(sent_clean,sec_str)['rougeLsum'].precision for sec_str in sec_strings] #changed recall to precision
        section_label = np.argmax(rouge_scores)
        abstract_groupings[section_label].append(sent)
    return abstract_groupings 
    
# for a given article d as dictionary we filter sections and mark abstracts
# output is a list of strings encoded as dictionaries, each of these is a "new" article
def generate_summaries(d):
    outputs = []
    section_names = d["section_names"]
    section_text = np.array(d["sections"])
    section_classifications = np.array([match_section(sec) for sec in section_names])
    new_section_text = []
    for group in keys:
        if group == "literature":
            new_section_text.append("")
            continue
        locs = np.where(section_classifications==group)[0]
        text = section_text[locs]
        merged_text = " ".join([" ".join(v) for v in text])
        if len(merged_text) <= 1:
            new_section_text.append("")
        else:
            new_section_text.append(merged_text)
    abstract = d["abstract_text"]
    #abstract_groupings = match_abstract(abstract,new_section_text)
    # print(abstract_groupings)
    output_section_text = [" ".join(new_section_text)]
    data_output = json.dumps({'abstract_text':abstract, 'sections': output_section_text})
    return data_output


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('infile', help='path to the jsonlines data')
    ap.add_argument('outfile', help='path to the output file')
    ap.add_argument('--num',type = int,default = -1, help ='number of articles to process')
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
            print("Article no:", i, " of ", num_articles,"\n")
            line = f.readline()
            if not line.strip():
                continue
            line = line.strip()
            data = json.loads(line)
            data_output = generate_summaries(data)
            with open(args.outfile,"a",encoding='utf-8') as out:
                out.write(data_output+"\n")
                out.close()
        f.close()
        out.close()
