from pandas import DataFrame
from IPython.display import display, HTML
import re
import spacy


def make_tok_df(doc, tok_filter_func=False):
    """Return a DataFrame showing attributes for each token in doc."""

    if tok_filter_func:
        toks = list(filter(tok_filter_func, doc))
    else:
        toks = doc
    doc_dict = {'tok_ent': [tok.ent_type_ for tok in toks],
                'toks': [tok for tok in toks],
                'lemma': [tok.lemma_ for tok in toks],
                'dep': [tok.dep_ for tok in toks],
                'head': [tok.head for tok in toks],
                'h_dep': [tok.head.dep_ for tok in toks],
                'dep_def': [spacy.explain(tok.dep_) for tok in toks],
                'pos': [tok.pos_ for tok in toks],
                'tag': [tok.tag_ for tok in toks],
                'tag_def': [spacy.explain(tok.tag_) for tok in toks],
                }
    columns = [ 'tok_ent', 'toks', 'lemma', 'dep', 'head', 'h_dep', 'pos', 'tag',  'dep_def', 'tag_def' ]
    return DataFrame(doc_dict, columns=columns)


# Generic function used in print_doc_info wrapper below
def make_span_df(doc, entities=True, span_filter_func=False):
    """Return df showing attributes for each entity or noun chunk in doc."""

    columns = ['tok_i', 'entity', 'ent_label', 'root', 'root_ent',
               'root_dep' ,'dep_def', 'root_head',  'root_head_dep',
               'root_head_pos'  ]
    if entities:
        target_spans = doc.ents
        df_name = 'doc_entities'
    else:
        target_spans = list(doc.noun_chunks)
        df_name = 'doc_noun_chunks'

    if span_filter_func:
        target_spans = list(filter(span_filter_func, target_spans))

    doc_dict = {'tok_i': [e.start for e in target_spans],
                'entity': [e.text for e in target_spans],
                'ent_label': [e.label_ for e in target_spans],
                'root': [e.root.text for e in target_spans],
                'root_ent': [e.root.ent_type_ for e in target_spans],
                'root_dep': [e.root.dep_ for e in target_spans],
                'dep_def': [spacy.explain(e.root.dep_) for e in target_spans],
                'root_head': [e.root.head for e in target_spans],
                'root_head_dep': [e.root.head.dep_ for e in target_spans],
                'root_head_pos': [e.root.head.pos_ for e in target_spans]}
    #    try :
    df = DataFrame(doc_dict, columns=columns)
    df.name = df_name
    if entities:
        df_cols = [x for x in df.columns.tolist() if x != 'root_ent']
        df = df[df_cols]
    else:
        df.columns = ['tok_i', 'noun_chunk', 'ent_label', 'root', 'root_ent', 'root_dep' ,'dep_def',
                      'root_head',  'root_head_dep', 'root_head_pos' ]
        df_cols = [x for x in df.columns.tolist() if x != 'ent_label']
        df = df[df_cols]
    #    except: 
    #        df = [(k, doc_dict[k]) for k in doc_dict.keys()]
    return df


# Print consecutive html dataframes in notebook
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


def make_fact_df(docs, re_func, nlp, df=0, id_fields=0, verbose=False):
    """Return data frame with rows for each fact extracted. """

    # Initialize list of fields to maintain from input df. 
    input_df_fields = ['doc_ids']

    # Turn doc into a list if a single string is passed
    if type(docs) == type(""):
        docs = [docs]

    if type(df) == type(DataFrame()):
        doc_ids = df.index.tolist()  # Use the index to identify documents
        if id_fields:
            input_df_fields = input_df_fields + id_fields
        docs = docs.tolist()

    else:
        doc_ids = list(range(len(docs)))

    err_ids = []
    all_docs, all_rels, doc_id_list = [], [], []
    for i, doc in enumerate(docs):
        try:
            doc_rels = re_func(nlp(doc))
        except:
            print("Error at index", i)
            err_ids = err_ids + [i]
            if len(err_ids) < 20:
                print("Error doc:")
                print(doc)

        if doc_rels:
            if all_rels:
                doc_list = [doc] * len(doc_rels)
                all_docs = all_docs + doc_list
                all_rels = all_rels + doc_rels
                doc_id_list = doc_id_list + [doc_ids[i]] * len(doc_rels)
            else:
                try:  # Get field names, if available
                    field_names = list(doc_rels[0]._fields)
                except:
                    field_names = list(range(len(doc_rels[0])))
                doc_list = [doc] * len(doc_rels)
                all_docs = all_docs + doc_list
                all_rels = all_rels + doc_rels
                doc_id_list = doc_id_list + [doc_ids[i]] * len(doc_rels)

    if all_rels:
        if verbose:
            print("len of all_rels is: " + str(len(all_rels)))
            print("len of all_docs is: " + str(len(all_rels)))
            print("len of doc_id_list is: " + str(len(doc_id_list)))
        fact_dict = {}
        for i, f in enumerate(field_names):
            fact_dict[f] = [rel[i] for rel in all_rels]

        fact_dict['doc_ids'] = doc_id_list
        output_df_fields = input_df_fields + field_names
        output_df = DataFrame(fact_dict, columns=output_df_fields)
        return output_df
    if err_ids:
        print("Error ids:")
        print(err_ids)
    return DataFrame(columns=['doc_ids', 'sent_num', 'word_num', 'subject', 'verb', 'quantity',
                                 'quantity_type', 'type_token', 'word', 'sentence'])


singles_word_list = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
teens_word_list = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
tens_word_list = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

tens_dict = {}
for w, n in zip([y + '-' for y in tens_word_list], list(range(2, 10))):
    tens_dict[w] = n
teens_dict = {}
for w, n in zip(teens_word_list, list(range(10, 20))):
    teens_dict[w] = n
singles_dict = {}
for w, n in zip(singles_word_list, list(range(1, 10))):
    singles_dict[w] = n


def add_units_and_values(df, quantity_col):
    """Create units, data_value columns from tokens in quantity_col."""
    if df.shape[0] == 0:
        return DataFrame(columns=df.columns.tolist() + ['units', 'data_value'])

    df = df.copy()
    df.loc[:, 'units'] = 'ones'

    # Identify columns with more than one token
    # Rows with more than one token for quantity contain either words as numbers
    # or non-numeric qualifiers
    quantity_bs = df.loc[:, quantity_col].apply(lambda x: len(x.text.split())) > 1

    # Remove tokens that are not numbers or number words
    df.loc[quantity_bs, 'units'] = df.loc[quantity_bs, quantity_col].apply(
        lambda x: [x for x in x if x._.is_num_word == True])

    empty_units_bs = df.units.apply(lambda x: len(x)) == 0
    not_empty_units = df.units.apply(lambda x: len(x)) == 1

    df.loc[empty_units_bs, 'units'] = 'ones'
    df.loc[not_empty_units, 'units'] = df.loc[not_empty_units, quantity_col].apply(
        lambda x: [x.text for x in x if x._.is_num_word == True][-1])

    # Initialize the values column
    df['data_value'] = '0'
    df.loc[~quantity_bs, 'data_value'] = df.loc[~quantity_bs, quantity_col].apply(lambda x: x.text)
    df.loc[quantity_bs, 'data_value'] = df.loc[quantity_bs, quantity_col].apply(
        lambda x: [x.text for x in x if x.like_num == True][0])

    comma_bs = df['data_value'].str.contains(",")

    # Create dictionary mappings for small number words
    df.replace({'data_value': singles_dict}, inplace=True)
    df.replace({'data_value': teens_dict}, inplace=True)

    df.loc[comma_bs, 'data_value'] = df.loc[comma_bs, 'data_value'].apply(lambda x: x.replace(',', ''))
    not_num_bs = df['data_value'].astype('str').str.contains(re.compile(r"[^0-9.]"))
    df.loc[not_num_bs, 'data_value'] = 0
    df.loc[comma_bs, 'data_value'] = df.loc[comma_bs, 'data_value'].apply(lambda x: float(x))
    df.loc[~comma_bs, 'data_value'] = df.loc[~comma_bs, 'data_value'].apply(lambda x: float(x))

    # Multiply values by 1000 if units = 'thousand'
    df.loc[df.units == 'thousand', 'data_value'] = df.loc[df.units == 'thousand', 'data_value'] * 1000

    return df


def fix_token_columns(df, col_list):
    """Return df with tokens converted to text."""
    if df.shape[0] == 0:
        return df
    df = df.copy()
    for col in col_list:
        df.loc[:, col] = df.loc[:, col].apply(lambda x: x.text)
    return df
