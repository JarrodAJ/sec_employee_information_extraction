def get_case_combos(str_list, fast=False):
    """Return a list with original, lower, upper, and title case."""
    
    if not fast: # Preserve some rational ordering
        case_combos = [s.lower() for s in str_list] + [s.upper() for s in str_list] 
        case_combos = case_combos + [s.title() for s in str_list if s.title() not in case_combos] 
        case_combos = case_combos + [s for s in str_list if s not in case_combos]
        return case_combos
    
    case_combos = str_list + [s.lower() for s in str_list] + [s.upper() for s in str_list] + [s.title() for s in str_list]
    return list(set(case_combos))

class EmpTypeRecognizer(object):
    """A spaCy v2.0 pipeline component that sets entity annotations
    based on list of terms. Terms are labelled as EMP_TYPE. Additionally,
    ._.has_emp_type and ._.is_emp_type is set on the Doc/Span and Token
    respectively."""
    name = 'employee_types'  # component name, will show up in the pipeline
    
    def __init__(self, nlp, terms_dict, label='EMP_TYPE'):
        """Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        self.label = nlp.vocab.strings[label]  # get entity label ID

        # Set up the PhraseMatcher – it can now take Doc objects as patterns,
        # so even if the list of terms is long, it's very efficient
        self.matcher = PhraseMatcher(nlp.vocab)
        for match_label in terms_dict.keys():
            patterns = [nlp(term) for term in terms_dict[match_label]]
            #patterns = [nlp(term) for term in terms]
            
            self.matcher.add(match_label, None, *patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        Token.set_extension('is_emp_type', default=False, force=True)
        Token.set_extension('is_part_time', default=False, force=True)
        Token.set_extension('is_full_time', default=False, force=True)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_emp_type == True.
        Doc.set_extension('has_emp_type', getter=self.has_emp_type, force=True)
        Span.set_extension('has_emp_type', getter=self.has_emp_type, force=True)
        Doc.set_extension('has_part_time', getter=self.has_part_time, force=True)
        Span.set_extension('has_part_time', getter=self.has_part_time, force=True)
        Doc.set_extension('has_full_time', getter=self.has_full_time, force=True)
        Span.set_extension('has_full_time', getter=self.has_full_time, force=True)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        for match_id, start, end in matches:
            # Generate Span representing the entity & set label
            entity = Span(doc, start, end, label=match_id)
            spans.append(entity)
            # Set custom attribute on each token of the entity
            for token in entity:
                token._.set('is_emp_type', True)
                if doc.vocab.strings[match_id] == 'PART_TIME':
                    token._.set('is_part_time', True)
                elif doc.vocab.strings[match_id] == 'FULL_TIME':
                    token._.set('is_full_time', True)
                    
            # Overwrite doc.ents and add entity – be careful not to replace!
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            # Iterate over all spans and merge them into one token. This is done
            # after setting the entities – otherwise, it would cause mismatched
            # indices!
            span.merge()
        return doc  # don't forget to return the Doc!

    def has_emp_type(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is an employee type. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_emp_type' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_emp_type') for t in tokens])
    
    def has_part_time(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is indicates part time. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_part_time' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_part_time') for t in tokens])
    
    def has_full_time(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is indicates full time. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_full_time' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_full_time') for t in tokens])

part_time_terms = get_case_combos(["half-time", "half time", "part-time", "part time"])
full_time_terms = get_case_combos(["full-time", "full time", "40-hour equivalent", "40 hour equivalent", "full-time equivalent", "full time equivalent", "full-"])
emp_type_dict = {'PART_TIME': part_time_terms, 
                'FULL_TIME': full_time_terms}

# Templated from: https://spacy.io/usage/processing-pipelines#custom-components 
class EmpNounRecognizer(object):
    """A spaCy v2.0 pipeline component that sets entity annotations
    based on list of terms. Terms are labelled as EMP_NOUN. Additionally,
    ._.has_emp_noun and ._.is_emp_noun is set on the Doc/Span and Token
    respectively."""
    name = 'employee_nouns'  # component name, will show up in the pipeline

    def __init__(self, nlp, terms=tuple(), label='EMP_NOUN'):
        """Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        self.label = nlp.vocab.strings[label]  # get entity label ID

        # Set up the PhraseMatcher – it can now take Doc objects as patterns,
        # so even if the list of terms is long, it's very efficient
        patterns = [nlp(term) for term in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('EMP_NOUN', None, *patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        Token.set_extension('is_emp_noun', default=False, force=True)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_emp_noun == True.
        Doc.set_extension('has_emp_noun', getter=self.has_emp_noun, force=True)
        Span.set_extension('has_emp_noun', getter=self.has_emp_noun, force=True)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        for _, start, end in matches:
            # Generate Span representing the entity & set label
            entity = Span(doc, start, end, label=self.label)
            spans.append(entity)
            # Set custom attribute on each token of the entity
            for token in entity:
                token._.set('is_emp_noun', True)
            # Overwrite doc.ents and add entity – be careful not to replace!
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            # Iterate over all spans and merge them into one token. This is done
            # after setting the entities – otherwise, it would cause mismatched
            # indices!
            span.merge()
        return doc  # don't forget to return the Doc!

    def has_emp_noun(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is an employee noun. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_emp_noun' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_emp_noun') for t in tokens])

emp_terms_list = ["associates", "employees", "equivalents", "FTEs", "FTE's", "headcount", "individuals", 
                  "people", "persons", "team members", "workers", "workforce"]
emp_terms_list = get_case_combos(emp_terms_list)

class NumberWordRecognizer(object):
    """A spaCy v2.0 pipeline component that sets entity annotations
    based on list of terms. Terms are labelled as NUM_WORD. Additionally,
    ._.has_num_word and ._.is_num_word is set on the Doc/Span and Token
    respectively."""
    name = 'number_words'  # component name, will show up in the pipeline

    def __init__(self, nlp, terms=tuple(), label='NUM_WORD'):
        """Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        self.label = nlp.vocab.strings[label]  # get entity label ID

        # Set up the PhraseMatcher – it can now take Doc objects as patterns,
        # so even if the list of terms is long, it's very efficient
        patterns = [nlp(term) for term in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('NUM_WORD', None, *patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        Token.set_extension('is_num_word', default=False, force=True)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_num_word == True.
        Doc.set_extension('has_num_word', getter=self.has_num_word, force=True)
        Span.set_extension('has_num_word', getter=self.has_num_word, force=True)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        
        matches = self.matcher(doc)
        spans = []  # collect the matched spans here

        for _, start, end in matches:
            spans.append(doc[start:end])
            # Set custom attribute on each token of the entity
        for span in spans:
            span.merge()
            for token in span:
                token._.set('is_num_word', True)
        return doc  

    def has_num_word(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a nubmer word. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_num_word' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_num_word') for t in tokens])

    
class YearMatcher(object):
    name = 'year_matcher'
    
    def __init__(self, nlp, pattern_list, match_id='Year'):
        # register a new token extension to flag year tokens
        Token.set_extension('is_year', default=False, force=True)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(match_id, None, pattern_list)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []  # collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        for span in spans:
            span.merge()   # merge
            for token in span:
                token._.is_year = True  # mark token as a year
        return doc

year_patterns = [{'ENT_TYPE': 'DATE', 'TAG' : 'CD', 'SHAPE' : 'dddd'}]

class FalseDateMatcher(object):
    """A spaCy pipeline component to flag arabic numbers if they 
    include commas or are greater than 31. Its main use is to 
    mitigate spaCy NER false positives."""
    name = 'false_date'
    
    regex_pat = re.compile(r"^([4-9][\d]|3[2-9]|(([0-9]{1,3},)*[0-9]{3}([.][0-9])?))$")
    
    def __init__(self, nlp, pattern_list, match_id='FALSE_DATE', label='FALSE_DATE', regex_pat = regex_pat):
        # register a new token extension to flag false_date tokens
        
        self.label = nlp.vocab.strings[label]  # get entity label ID
        self.orig_label = nlp.vocab.strings['DATE']  # get entity label ID for date
        Token.set_extension('is_false_date', default=False, force=True)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(match_id, None, pattern_list)
        self.regex_pat = regex_pat
        
        
         # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_false_date == True.
        Doc.set_extension('has_false_date', getter=self.has_false_date, force=True)
        Span.set_extension('has_false_date', getter=self.has_false_date, force=True)

    def __call__(self, doc):
        matches = self.matcher(doc)
        candidate_spans = []  # collect the matched spans here
        spans = [] # for tokens that match regex
        for match_id, start, end in matches:
            candidate_spans.append(doc[start:end])
        for span in candidate_spans:
#            span.merge()   # merge
            for token in span:
                if re.match(self.regex_pat, token.text):
                    # Generate Span representing the entity & set label
                    entity = Span(doc, token.i, token.i + 1, label=self.label)
                    spans.append(entity)
                    token._.is_false_date = True  # mark token as a false date
                    # Get original date span
                    orig_span = [e for e in doc.ents if token in e][0]
                     # Create ents list to add to doc ents
                    new_ents = []
                    # re-run NER on rest of span
                    if token.i > orig_span.start:
                        left_span = doc[orig_span.start : token.i]
                        left_ents = list(nlp(left_span.text).ents)
                        if left_ents: 
                            new_ents.append(Span(doc, left_span.start, left_span.end, label=self.orig_label))
                    new_ents.append(entity)
                    if token.i < orig_span.end:
                        right_span = doc[token.i + 1 : orig_span.end + 1]
                        right_ents = list(nlp(right_span.text).ents) 
                        if right_ents:
                            new_ents.append(Span(doc, right_span.start, right_span.end, label=self.orig_label))
                    
                     # Overwrite doc.ents and add entity – be careful not to replace!
                    #doc.ents = list(doc.ents) + [entity]
                    doc.ents = list(doc.ents) + new_ents
            for span in spans:
            # Iterate over all spans and merge them into one token. This is done
            # after setting the entities – otherwise, it would cause mismatched
            # indices!
                span.merge()
            
        return doc
    
    def has_false_date(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a false date. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_false_date' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_false_date') for t in tokens])

false_date_patterns = [{'ENT_TYPE': 'DATE', 'TAG' : 'CD'}]

singles_word_list = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
teens_word_list = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
tens_word_list = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
magnitude_word_list = ["hundred", "thousand", "million", "billion"]

teen_unit_combos = [x.join([y,z])  for x in [" ", "-"] for y in get_case_combos(tens_word_list) for z in get_case_combos(singles_word_list) ]
number_word_list = get_case_combos(singles_word_list) + get_case_combos(teens_word_list) + get_case_combos(tens_word_list) + get_case_combos(magnitude_word_list) + teen_unit_combos

tens_dict = {}
for w, n in zip([y + '-'  for y in tens_word_list ], list(range(2,10))):
    tens_dict[w] = n
teens_dict = {}
for w, n in zip(teens_word_list, list(range(10,20))):
    teens_dict[w] = n
singles_dict = {}
for w, n in zip(singles_word_list, list(range(1,10))):
    singles_dict[w] = n

nlp = spacy.load('en_core_web_lg')

emp_noun_recognizer = EmpNounRecognizer(nlp, emp_terms_list)
nlp.add_pipe(emp_noun_recognizer, last=True) 

emp_type_recognizer = EmpTypeRecognizer(nlp, emp_type_dict)
nlp.add_pipe(emp_type_recognizer, last=True) 

number_word_recognizer = NumberWordRecognizer(nlp, number_word_list)
nlp.add_pipe(number_word_recognizer, last=True) 

year_matcher = YearMatcher(nlp, year_patterns)
nlp.add_pipe(year_matcher, last=True) 

false_date_matcher = FalseDateMatcher(nlp, false_date_patterns)
nlp.add_pipe(false_date_matcher, last=True) 