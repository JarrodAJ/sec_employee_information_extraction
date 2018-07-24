import spacy;
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span, Token
import pandas as pd
import re

from path import Path, getcwdu
from collections import namedtuple
from IPython.display import display, HTML

def make_tok_df(doc, tok_filter_func=False):
    """Return a dataframe showing attributes for each token in doc."""
    
    if tok_filter_func:
        toks = list(filter(tok_filter_func, doc))
    else:
        toks = doc
    doc_dict = {'tok_ent' : [tok.ent_type_ for tok in toks], 
        'toks' : [tok for tok in toks], 
        'lemma' : [tok.lemma_ for tok in toks], 
        'dep' : [tok.dep_ for tok in toks], 
        'head' : [tok.head for tok in toks], 
        'h_dep' : [tok.head.dep_ for tok in toks], 
        'dep_def' : [spacy.explain(tok.dep_) for tok in toks],
        'pos' : [tok.pos_ for tok in toks], 
        'tag' : [tok.tag_ for tok in toks], 
        'tag_def' : [spacy.explain(tok.tag_) for tok in toks], 
               }
    columns = [ 'tok_ent', 'toks', 'lemma', 'dep', 'head', 'h_dep', 'pos', 'tag',  'dep_def', 'tag_def' ]
    return pd.DataFrame(doc_dict, columns=columns)


def make_span_df(doc, entities=True, span_filter_func=False):
    """Return df showing attributes for each entity or noun chunk in doc."""
    
    columns = ['tok_i', 'entity', 'ent_label', 'root', 'root_ent', 
               'root_dep','dep_def', 'root_head',  'root_head_dep', 
               'root_head_pos'  ]
    if entities:
        target_spans = doc.ents
        df_name = 'doc_entities'
    else: 
        target_spans = list(doc.noun_chunks)
        df_name = 'doc_noun_chunks'
        
    if span_filter_func:
        target_spans = list(filter(span_filter_func, target_spans))
        
    doc_dict = {'tok_i' : [e.start for e in target_spans], 
        'entity' : [e.text for e in target_spans], 
        'ent_label' : [e.label_ for e in target_spans], 
        'root' : [e.root.text for e in target_spans], 
        'root_ent' : [e.root.ent_type_ for e in target_spans], 
        'root_dep' : [e.root.dep_ for e in target_spans], 
        'dep_def' : [spacy.explain(e.root.dep_) for e in target_spans],
        'root_head' : [e.root.head for e in target_spans], 
        'root_head_dep' : [e.root.head.dep_ for e in target_spans], 
        'root_head_pos' : [e.root.head.pos_ for e in target_spans]}
#    try :
    df = pd.DataFrame(doc_dict, columns=columns) 
    df.name = df_name
    if entities:
        df_cols = [x for x in df.columns.tolist() if x != 'root_ent']
        df = df[df_cols]
    else:
        df.columns = ['tok_i', 'noun_chunk', 'ent_label', 'root', 'root_ent', 'root_dep','dep_def',
                      'root_head',  'root_head_dep', 'root_head_pos' ]
        df_cols = [x for x in df.columns.tolist() if x != 'ent_label']
        df = df[df_cols]
#    except: 
#        df = [(k, doc_dict[k]) for k in doc_dict.keys()]
    return df

def print_df(df, print_df_name=True):
    """Display rendered HTML version of df in Jupyter notebook"""
    
    if print_df_name and hasattr(df, 'name'):
        print("DataFrame is named " + str(df.name))
    display(HTML(df.to_html()))


def print_doc_info(doc):
    """Print dfs for doc entities, noun_chunks, and CARDINAL toks"""
    
    print("doc is: ")
    print(doc)
    print('-' * 50)
    print("Entities are: ")
    print_df(make_span_df(doc), print_df_name=False)
    print('-' * 50)
    print("Noun chunks are: ")
    print_df(make_span_df(doc, entities=False), print_df_name=False)
    print('-' * 50)
    print("Cardinal entities are: ")
    card_filter = lambda w: w.ent_type_ == 'CARDINAL'
    print_df(make_tok_df(doc, card_filter), print_df_name=False)

def find_root_tok(tok):
    """Return (tok's root node, num steps to reach root)."""
    
    steps = 0 
    root_tok = tok
    while root_tok.dep_ != 'ROOT':
        steps +=1 
        root_tok = root_tok.head
    return (root_tok, steps)

def find_verb_tok(tok):
    """Return first verb ancestor of tok."""
    verb_tok = 0
    for a in tok.ancestors:
        if a.pos_ == 'VERB' and a.dep_ in ['ROOT', 'ccomp']:
            return a
    return verb_tok
    
def find_tok_side_of_root(tok, root_tok):
    """Return 'right' or 'left' if tok is in subtree of root."""
    
    for a in [tok] + list(tok.ancestors): # The ancestors of a token will either be in root.rights or root.lefts
        if a in root_tok.lefts:
            return 'left'
        elif a in root_tok.rights:
            return 'right'
    else:
        return None

def find_subject(root_tok, verbose=False):
    """Return list of nominal subject"""
    subjects = [w for w in root_tok.lefts if w.dep_ == 'nsubj']
    try:
#        for i, s in enumerate(subjects):
#            print("nsubj "+str(i)+" of " + str(root_tok) + " is : " + str(s))
        return subjects[0]
    except:
        if verbose == True:
            print("No nsubj found left of ROOT. Noun phrases left of root are:")
            print([x for x in list(root_tok.doc.noun_chunk)])
        return False
        
def get_org_span(tok):
    """Return the entity span if token has ORG ent_type_."""
    if tok.ent_type_ == 'ORG':
        subject = [e for e in tok.doc.ents if tok in e][0]
        return subject
    return tok

def check_emp_type_flags(tok):
    """Return 'Part-Time' or 'Full-Time' if corresponding flags
    are set to True."""
    
    if tok._.is_part_time == True:
        return 'Part-Time Employees'
    if tok._.is_full_time == True:
        return 'Full-Time Employees'
    return 'Other Employees'

def find_emp_type_tok(tok):
    """Return child token left of tok if emp_type flagged or ADJ."""
    
    flagged_toks = [t for t in tok.children if t._.is_emp_type == True]
    if flagged_toks:
        return flagged_toks[0]
    
    candidate_tok = tok.doc[tok.i - 1]  
    while candidate_tok.is_punct == True:
        candidate_tok = tok.doc[candidate_tok.i - 1]
    if candidate_tok.head == tok:
        if candidate_tok.pos_ == 'ADJ' or candidate_tok.dep_ == 'compound':
            return candidate_tok
    return 0

def get_nummod_tok(tok, years, verbose=False):
    """Return tok.children that are nummod and card entities."""
    
    num_toks = [c for c in tok.children if c.dep_ == 'nummod' and c.ent_type_ == 'CARDINAL']
    if num_toks:
        if verbose == True:
            print("Num_toks are: " + str(num_toks))
        num_tok = num_toks[0]
        num_tok_conj = [c for c in num_tok.children if c.dep_=='conj' and c.tag_ == 'CD']
        
        if num_tok_conj:
            if verbose == True:
                print("num_tok has conjugate children:" + str(num_tok_conj))
                print("num_tok subtree is :" + str(list(num_tok.subtree)))
            cards = [(c.i, c) for c in num_tok.subtree if c.tag_== 'CD' and c.ent_type_ == 'CARDINAL']
            
            if len(years) == len(cards):
                order_indices = [years.index(y) for y in sorted(years, reverse=True, key = lambda x: x[1])]
                #year_emps = [(years[i][1].text, cards[i][1].text) for i in order_indices]
                #num_tok = max(year_emps)[1]
                year_emps = sorted([(years[i][1], cards[i][1]) for i in order_indices], reverse=True, key = lambda x: x[0].text)
                num_tok = year_emps[0][1]
                if verbose == True:
                    print("years: " + str(years))
                    print("cards: " + str(cards))
                    print("order_indices: " + str(order_indices))
                    print("year_emps: " + str(year_emps))
        return num_tok
    
    return 0      

def extract_emp_relations(doc, verb_list=False, verbose=False):
    """Return tuple of extracted relations."""
    if not verb_list:
        verb_list = ['be', 'employ', 'have']
    
    relation_tuples = []
    
    tuple_field_names = ["sent_num", "word_num", "subject", "verb", "quantity", "quantity_type", "type_token" , "word", "sentence"]
    RelationDetails = namedtuple('RelationDetails', tuple_field_names)
    
    for sent_id, sent in enumerate(doc.sents):
        
        # Find the root token
        root_tok, depth = find_root_tok(sent[0])
        
        match_pairs = []
        num_tok, num_tok_conj, subject, year_conj = (False, False, False, False)
        years = [(y.i, y) for y in root_tok.subtree if y._.is_year == True] # Need to change to root.subtree to only return the word's sentence

        for word_id, word in enumerate(filter(lambda w: w.ent_type_ == 'EMP_NOUN', sent)):  
            
            if verbose == True:
                print("Word_id is : " + str(word_id))
                print("Word is : " + str(word))
            
            # Find first verb ancestor 
            verb_tok = find_verb_tok(word)
            if not verb_tok:
                continue
            
            # If verb does not have expected lemma, move to next sentence
            if root_tok.lemma_ not in verb_list:
                root_tok = verb_tok
                if verbose == True:
                        print("Root token lemma not one of ['be', 'employ', 'have']. ")
                        print("Root token, lemma are : " + str(root_tok) + " " + str(root_tok.lemma_))
                        print(list(root_tok.subtree))
                
                if verb_tok.lemma_ not in verb_list:
                    if verbose == True:
                        print("verb token lemma not one of ['be', 'employ', 'have']. ")
                        print("verb token, lemma are : " + str(verb_tok) + " " + str(verb_tok.lemma_))
                        print(list(verb_tok.subtree))
                    continue

            emp_type_tok = find_emp_type_tok(word)
            emp_type = 'Other Employees'
            if emp_type_tok:
                emp_type = check_emp_type_flags(emp_type_tok)
            parts_found = []
            # Find out if the employee noun is in subject (left) or predicate (right)
            left_side = []; right_side = []
            emp_tok_side = find_tok_side_of_root(word, root_tok)
            if emp_tok_side == 'left':
                left_side.append(word)
            elif emp_tok_side == 'right':
                right_side.append(word)
            else:
                if verbose == True:
                    print("No ancestor of'" + str(word) + "' is in root.rights or root.lefts.")    

            if verbose == True:
                print("Dep_ of EMP_NOUN is: " + str(word.dep_))
            if word.dep_ in ('attr', 'dobj', 'compound') or word.dep_ == 'pobj' and word.head.dep_ == 'prep':
                num_tok = get_nummod_tok(word, years, verbose = verbose)      
                if num_tok:
                    match_pairs.append((num_tok, word))
                else:
                    cards = [e for e in word.doc.ents if e.label_ == 'CARDINAL' and e.root in root_tok.rights]
                    if cards:
                        cards = cards + [c for c in word.doc.ents if c.root in cards[0].root.subtree and c not in cards and c.label_ == 'CARDINAL']           
                        if word in left_side:    
                            if len(years) > 0:                       
                                emp_counts = [(c.start, c) for c in sorted(cards, reverse=False, key = lambda x: x.start)]                      
                                order_indices = [years.index(y) for y in sorted(years, reverse=True, key = lambda x: x[1])]
                                try: 
                                    #year_emps = [(years[i][1].text, emp_counts[i][1].text) for i in order_indices]
                                    year_emps = sorted([(years[i][1], emp_counts[i][1]) for i in order_indices], reverse=True, key = lambda x: x[0].text)
                                    if verbose == True:
                                        print("years: " + str(years))
                                        print("emp_counts: " + str(emp_counts))
                                        print("order_indices: " + str(order_indices))
                                        print("year_emps: " + str(year_emps))
                                    #num_tok = max(year_emps)[1]
                                    num_tok = year_emps[0][1]
                                except:
                                    print(str("==" * 20))
                                    print("Length of emp_counts is : " + str(len(emp_counts)) + 
                                         " while length of years is : " + str(len(years)))
                                    num_tok = cards[0]
                                    print(str("-" * 20))
                                    print(word.doc.text)
                                if verbose == True:
                                    print("Sentence has multiple years:" + str(years))
                                    print("First card subtree is :" + str(list(cards[0].subtree)))
                                    print("years: " + str(years))
                                    print("cards: " + str(cards))
                                    print("emp_counts: " + str(emp_counts))
                    #                print("order_indices: " + str(order_indices))                   
                                match_pairs.append((num_tok, word))
                        else:
                            if verbose == True:
                                print("Emp_tok is in right side; appending first card.")
                            match_pairs.append((cards[0], word))
                    elif verb_tok.dep_ == 'relcl':
                        cards = [e for e in word.doc.ents if e.label_ == 'CARDINAL' and e.root in verb_tok.lefts]
                        if cards:
                            match_pairs.append((cards[0], word))
                            num_tok = cards[0]
                if verbose == True:
                    print("Root is at "+str(depth)+" steps from "+str(word)+".")
                subject = find_subject(root_tok)
                if not subject: # For debugging
                    if verbose == True:
                        print("No nsubj found left of ROOT. Noun phrases left of root are:")
                        left_filter = lambda e: e.root in root_tok.lefts
                        print_df(make_span_df(doc, entities=False, span_filter_func=left_filter))
                else:
                    if subject == word.head.head: # If word is part of prep phrase of subject
                        subject = doc[subject.left_edge.i : subject.right_edge.i + 1]
                    else: 
                        subject = get_org_span(subject) # Use full span of ORG entity if subject tok is in ORG 
                    parts_found.append(subject)
                    match_pairs.append((subject, word))
                    #[print(str(p) + '  :  ' + str(p.dep_)) for p in subject.subtree]
                    sub_poss = [p for p in subject.subtree if p.dep_ == 'poss']
    #                if sub_poss:
    #                    sub_poss = sub_poss[0]
    #                    match_pairs.append((sub_poss, word))
                    if root_tok:
                        parts_found.append(root_tok)
                    if num_tok:
                        parts_found.append(num_tok)
                        parts_found.append(emp_type)
                        parts_found.append(emp_type_tok)
                        parts_found.append(word)
                    elif word.head.head.head.pos_ == 'VERB':
                        if verbose == True:
                            print("No num_tok. ")
                        cards = [c for c in word.head.head.head.rights if c.tag_== 'CD' and c.ent_type_ == 'CARDINAL']
                        years = [(y.i, y) for y in root_tok.subtree if y._.is_year == True]
                        match_pairs.append((years, cards))
                        if cards:
                            match_pairs.append((cards[0], word))

            elif word.dep_ == 'conj':
                num_tok = get_nummod_tok(word, years, verbose = verbose)
                head_num_tok = [w for w in [word.head] if w.tag_ == 'CD' and w.ent_type_ == 'CARDINAL']            
                if verbose == True:
                    print("Emp_noun token has dep_ == 'conj'.")
                    print("Child num_tok: " + str(num_tok))
                    print("Head num_tok: " + str(head_num_tok))
                if num_tok and head_num_tok:
                    if verbose == True:
                        print("child_num_tok and head_num_toks")                      
                    num_toks = [num_tok] + head_num_tok
                    years = [(y.i, y) for y in root_tok.subtree if y._.is_year == True]
                    if verbose == True:
                        print("years: " + str(years))   
                        print("num_toks: " + str(num_toks))   
                    if head_num_tok[0].dep_ == 'conj' or head_num_tok[0].head.ent_type_ == 'CARDINAL' :
                        num_toks = num_toks + [w for w in [head_num_tok[0].head] if w.tag_ == 'CD']
                    if len(years) > len(num_toks):
                        if doc[head_num_tok[0].i - 2].ent_type_ == 'CARDINAL':
                            num_toks = num_toks + [doc[head_num_tok[0].i - 2]] 
                        head_num_conjucts = [c for c in head_num_tok[0].conjuncts if c.tag_ == 'CD']
                        if head_num_conjucts:
                            if head_num_conjucts[0].ent_type_ != 'CARDINAL':
                                print(str("==" * 30))
                                print("Potential series token :" + str(head_num_conjucts[0]) + 
                                     " does not have CARDINAL entity type. ")
                                print("Entity type is: " + str(head_num_conjucts[0].ent_type_))
                                print("Token index: " + str(head_num_conjucts[0].i))
                                print("Doc is: " + str(word.doc))
                            num_toks = num_toks + head_num_conjucts
                    emp_counts = sorted([(c.i, c) for c in num_toks], key = lambda x: x[0])
                    if verbose == True:
                        print("emp_counts: " + str(emp_counts))
                    order_indices = [years.index(y) for y in sorted(years, reverse=True, key = lambda x: x[1])]
                    if verbose == True:
                        print("order_indices: " + str(order_indices))
                    try:
                        year_emps = sorted([(years[i][1], emp_counts[i][1]) for i in order_indices], reverse=True, key = lambda x: x[0].text)
                    except:
                        print(str("==" * 20))
                        print("Error on doc:")
                        print(str("-" * 20))
                        print(word.doc.text)
                        print(str("-" * 20))
                        print("Error sentence:")
                        print(sent)
                        
                    if verbose == True:
                        print("year_emps: " + str(year_emps))
                    #num_tok = max(year_emps)[1]
                    num_tok = year_emps[0][1]
                match_pairs.append((num_tok, word))
                
                if not subject:
                    try:
                        subject = get_org_span(find_subject(root_tok))
                    except:
                        continue
                parts_found.append(subject)
                parts_found.append(root_tok)
                parts_found.append(num_tok)
                parts_found.append(emp_type)
                parts_found.append(emp_type_tok)
                parts_found.append(word)
            
            else:
                continue
            
            if all([subject, root_tok, num_tok, emp_type ]):
                if verbose == True:
                    print(tuple(parts_found))
                details = [sent_id, word_id] + parts_found + [sent.text]
                relation_tuples.append(RelationDetails(*details))        
    return relation_tuples
    
def make_fact_df(docs, re_func, df=0, id_fields=0, verbose = False):
    """Return data frame with rows for each fact extracted. """
    
    # Initialize list of fields to maintain from input df. 
    input_df_fields = ['doc_ids']
    
    # Turn doc into a list if a single string is passed
    if type(docs) == type(""):
        docs = [docs]
    
    if type(df) == type(pd.DataFrame()) :
        doc_ids = df.index.tolist() # Use the index to identify documents
        if id_fields:
            input_df_fields = input_df_fields + id_fields 
        docs = docs.tolist()

    else: 
        doc_ids = list(range(len(docs)))
        
    all_docs, all_rels, doc_id_list = [], [], []
    for i, doc in enumerate(docs):
        doc_rels = re_func(nlp(doc))
        if doc_rels:         
            if all_rels:
                doc_list = [doc] * len(doc_rels)
                all_docs = all_docs + doc_list
                all_rels = all_rels + doc_rels
                doc_id_list = doc_id_list + [doc_ids[i]] * len(doc_rels)
            else: 
                try: # Get field names, if availble
                    field_names = list(doc_rels[0]._fields)
                except:
                    field_names = list(range(len(doc_rels[0])))
                doc_list = [doc] * len(doc_rels)
                all_docs = all_docs + doc_list     
                all_rels = all_rels + doc_rels
                doc_id_list = doc_id_list + [doc_ids[i]] * len(doc_rels)
           
    if all_rels:
        if verbose == True:
            print("len of all_rels is: " + str(len(all_rels)))
            print("len of all_docs is: " + str(len(all_rels)))
            print("len of doc_id_list is: " + str(len(doc_id_list)))
        fact_dict = {} 
        for i, f in enumerate(field_names):
            fact_dict[f] = [rel[i] for rel in all_rels]
        
        fact_dict['doc_ids'] = doc_id_list
        output_df_fields = input_df_fields + field_names
        output_df = pd.DataFrame(fact_dict, columns=output_df_fields)
        return output_df
    return 0

def add_units_and_values(df, quantity_col):
    """Create units, data_values columns from tokens in quantity_col."""
    df = df.copy()
    df.loc[:, 'units'] = 'ones'
    
    # Identify columns with more than one token
    # Rows with more than one token for quantity contain either words as numbers 
    # or non-numeric qualifiers
    quantity_bs = df.loc[:, quantity_col].apply(lambda x: len(x.text.split())) > 1
    
    # Remove tokens that are not numbers or number words
    df.loc[quantity_bs, 'units'] = df.loc[quantity_bs, quantity_col].apply(lambda x: [x for x in x if x._.is_num_word == True])
    
    
    empty_units_bs = df.units.apply(lambda x: len(x)) == 0
    not_empty_units= df.units.apply(lambda x: len(x)) == 1

    df.loc[empty_units_bs, 'units'] = 'ones'
    df.loc[not_empty_units, 'units'] = df.loc[not_empty_units, quantity_col].apply(lambda x: [x.text for x in x if x._.is_num_word == True][-1])
    
    # Initialize the values column
    df['data_values'] = '0'
    df.loc[~quantity_bs, 'data_values'] = df.loc[~quantity_bs, quantity_col].apply(lambda x: x.text)
    df.loc[quantity_bs, 'data_values'] = df.loc[quantity_bs, quantity_col].apply(lambda x: [x.text for x in x if x.like_num == True][0])  

    comma_bs = df['data_values'].str.contains(",")
    
    # Create dictionary mappings for small number words
    df.replace({'data_values' : singles_dict}, inplace=True)
    df.replace({'data_values' : teens_dict}, inplace=True)

    df.loc[comma_bs,'data_values'] = df.loc[comma_bs,'data_values'].apply(lambda x: x.replace(',', ''))
    not_num_bs = df['data_values'].astype('str').str.contains(re.compile(r"[^0-9.]"))
    df.loc[not_num_bs, 'data_values'] = 0
    df.loc[comma_bs, 'data_values'] = df.loc[comma_bs,'data_values'].apply(lambda x: float(x))
    df.loc[~comma_bs,'data_values'] = df.loc[~comma_bs,'data_values'].apply(lambda x: float(x))
    
    # Multiply values by 1000 if units = 'thousand'
    df.loc[df.units == 'thousand', 'data_values'] = df.loc[df.units == 'thousand', 'data_values'] * 1000
    
    return df
def fix_token_columns(df, col_list):
    """Return df with tokens converted to text."""
    df = df.copy()
    for col in col_list:
        df.loc[:,col] = df.loc[:, col].apply(lambda x: x.text)
    return df