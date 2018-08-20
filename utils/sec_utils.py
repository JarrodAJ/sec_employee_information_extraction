from bs4 import BeautifulSoup as bs
import copy
import glob
import os
from pathlib import PurePath
import pandas as pd
import re
import requests

filing_suffix, filings_dir = (".html",  PurePath('../employee_filings/').as_posix() + "/")
html_path_list = [PurePath(os.getcwd()).joinpath(file).as_posix() for file in glob.iglob("*".join([filings_dir, filing_suffix]))]
html_file_list = [PurePath(path).name for path in html_path_list]

head_block_re = re.compile(r"^(p|div|h[1-6])$")  # Identify block-level elements, aside from tables


def path_handler(path):
    """Return request text if path is url; otherwise returns file context.
    Automatically recognizes Edgar URLs.
    """

    edgar_link_pat = re.compile(
        r"(?:https://www.sec.gov/)(?:[a-z0-9/-]+)([0-9]{10}-?[0-9]{2}[-]?[0-9]{6})(?:[a-z0-9/-]*[.]+[a-z]+)", re.I)
    if re.match(edgar_link_pat, path):
        r = requests.get(path)
        r_html = r.text
        return r_html

    with open(path, encoding="utf8") as file:
        file_html = file.read()
        return file_html


def get_filing_info(accession_id):
    """Return Edgar (url, filing_name, filing_date, company) for Accession Id"""

    search_url = "".join(["https://searchwww.sec.gov/EDGARFSClient/jsp/EDGAR_MainAccess.jsp?search_text=",
                          accession_id,
                          "&isAdv=false"])
    search_html = requests.get(search_url).text

    edgar_trs = bs(search_html, 'lxml').find(
        "table", {'xmlns:autn': 'http://schemas.autonomy.com/aci/'}).find_all(
        "tr", {'class': None})
    try:
        js_info_pat = re.compile(r"javascript:open[a-z]*[(](.*)[)][;]", re.I)
        filing_result = [tr.find("a", id="viewFiling") for tr in edgar_trs if
                         tr.find_next_sibling("tr", {"class": "infoBorder"}).findChild("a", {
                             "title": "Parent Filing"}) is None]
        if filing_result:
            edgar_link_tag = filing_result[0]
        else:
            edgar_link_tag = edgar_trs[0].find_next("a", {"title": "Parent Filing"})
        href_list = re.search(js_info_pat, edgar_link_tag['href']).group(1).split(',')[:2]
        filing_name = edgar_link_tag.get_text(strip=True)
        filing_date = edgar_link_tag.parent.previous_sibling.i.get_text()
        url, company = href_list[0], href_list[1]
        return url, filing_name, filing_date, company
    except:
        print("Error retrieving url for input", accession_id)


def extract_paragraph_df(path_list, regex_list, header_regex, header_raw_regex,
                       head_block_regex=head_block_re, table_df=True):
    """Return dict with df of paragraphs that match the given regex.
    Optionally return df from matching tables as well."""

    acc_id_pat = re.compile(r"[0-9]{10}-?([0-9]{2})[-]?([0-9]{6})")
    block_re = re.compile(r"^(p|div|table)$")
    acc_id_list = []  # Replace with generic ID list
    para_list_orig = []
    tag_list = []
    emp_head_list = []
    emp_head_first_list = []
    tbl_acc_id_list = []
    tbl_tag_list = []
    for i, fl in enumerate(path_list):
        acc_id_check = re.search(acc_id_pat, fl)
        if acc_id_check:
            acc_id = acc_id_check.group(0)
            filing_year = "20" + acc_id_check.group(1)
        else:
            acc_id = str(i)
        tag_set = set()
        file_html = path_handler(fl)
        soup = bs(file_html, 'lxml')
        #        emp_head_flag = False
        #        emp_head_first_match = False
        if re.search(header_raw_regex, file_html):
            for ihead, hblock in enumerate(soup.body.find_all(string=header_regex, limit=4)):
                try:
                    emp_head_tag = hblock.find_parent(name=head_block_regex)
                    if emp_head_tag.name != 'table' and emp_head_tag.find_parent('table') is None:
                        emp_head_matched = False
                        #            print(emp_head_tag.name) ;print(emp_head_tag)
                        for i2, block in enumerate(emp_head_tag.find_next_siblings(block_re, limit=6)):
                            if block.find(string=[regex_list]) is not None and block.name != 'table':
                                block_tag = copy.copy(block)
                                if block_tag not in tag_set:
                                    acc_id_list.append(acc_id)
                                    tag_list.append(block_tag)
                                    para_list_orig.append(block_tag.get_text())
                                    tag_set.add(block_tag)
                                    emp_head_list.append(True)
                                    if not emp_head_matched:
                                        emp_head_flag = True
                                        emp_head_matched = True
                                        emp_head_first_list.append(True)
                                    else:
                                        emp_head_first_list.append(False)
                            if block.find('table') is not None:
                                #                    print('Found table match!')
                                tbl_block_tag = copy.copy(block)
                                if tbl_block_tag not in tag_set:
                                    tbl_acc_id_list.append(acc_id)
                                    tbl_tag_list.append(tbl_block_tag)
                                    tag_set.add(tbl_block_tag)
                except:
                    continue
        #        else:
        #            print('No Employees header')
        soup_emp_count = soup.body.find_all(string=[regex_list])
        soup_emp_paras = [x.find_parent(name=block_re) for x in soup_emp_count]
        soup_emp_paras = [x for x in soup_emp_paras if x is not None]
        for i2, block in enumerate(soup_emp_paras):
            #                print('Para number: ' + str(i2)); print(block)
            block_tag = copy.copy(block)
            if block_tag not in tag_set:
                if block.find('table') is not None:
                    tbl_acc_id_list.append(acc_id)
                    tbl_tag_list.append(block_tag)
                    tag_set.add(block_tag)
                else:
                    acc_id_list.append(acc_id)
                    tag_list.append(block_tag)
                    tag_set.add(block_tag)
                    para_list_orig.append(block_tag.get_text())
                    emp_head_list.append(False)
                    emp_head_first_list.append(False)

    paragraph_dict = {'acc_id': acc_id_list,
                        'para_text': [p.replace('\n', ' ').strip().replace(' ,', ',') for p in para_list_orig],
                        'len': [len(p) for p in para_list_orig],
                        'emp_header': emp_head_list,
                        'first_emp_head_block': emp_head_first_list,
                        'para_text_orig': para_list_orig,
                        'para_tag': tag_list,
                        'split': 'train',
                        'label': 0}
    # paragraph_input_df['para_text'] = paragraph_input_df.para_text_orig.replace('\n', ' ')
    p_columns = ['acc_id', 'para_text', 'len', 'emp_header', 'first_emp_head_block', 'para_text_orig',
                 'para_tag', 'split', 'label']

    paragraph_df = pd.DataFrame(paragraph_dict, columns=p_columns)

    if table_df:
        # Make DataFrame from <table> elements
        tbl_df = pd.DataFrame(data={'acc_id': tbl_acc_id_list,
                                    'tbl_html': tbl_tag_list,
                                    'split': 'train'})

        return {'paragraphs': paragraph_df,
                'tables': tbl_df}

    return {'paragraphs': paragraph_df}


if __name__ == "__main__":
    train_accession_ids = pd.read_csv('../data/train_accession_ids.csv', names=['acc_id'])['acc_id'].tolist()
    val_accession_ids = pd.read_csv('../data/val_accession_ids.csv', names=['acc_id'])['acc_id'].tolist()

    # Initial components
    employee_terms = "(associates|employees|full[ -]time[ -]equivalent(s)?|staff|team members|workers)"
    person_terms = "(individuals|people|persons)"  # These need additional cues
    workforce_terms = "(((employee|employment|head|personnel|staff|worker|workforce) (count(s)|level(s)|total(s))+)|(head-count|headcount|workforce))"
    employee_type_terms = "(full time|full-time|permanent|part time|part-time|regular|seasonal|temporary|total)"

    numeral_pat = "(([0-9]{1,3},)*[0-9]{1,3}([.][0-9])?)"  # Include numerals, requiring comma separation when appropriate, and allowing for decimals.
    rel_qualifiers = "(a total of|approximately|in aggregate|in total|(an|the) equivalent of|total)"

    space_pat = "( |\n)"  # In html, a space is often missing if the paragraph continues on the next line

    magnitude_words = "(hundred|thousand|million|billion)"
    num_words = "(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"

    # Composites to allow numerals and/or number words with relative modifiers
    num_pat = "".join(["((", numeral_pat, "|", num_words, ")(", space_pat, magnitude_words, ")*", ")"])
    num_emps_pat = "".join(
        [num_pat, space_pat, "(", rel_qualifiers, space_pat, ")*", "(", employee_type_terms, space_pat, ")*"])

    # Actual patterns to be used, named by the format they're meant to capture
    number_employees_pat = "".join(
        [num_emps_pat, employee_terms])  # A number followed by an employee term, allowing for qualifiers
    employed_num_pat = "".join(["employ((ed|s)?)?", space_pat, "(", rel_qualifiers, space_pat, ")*",
                                num_emps_pat])  # The verb employed, followed by a number
    emp_type_emp_term_pat = "".join(
        [employee_type_terms, space_pat, employee_terms])  # Part or full-time term, followed by an employee term
    employed_end_span_pat = "".join(["employed(", space_pat, rel_qualifiers,
                                     ")*$"])  # The actual number is often cut off by a <span/> element in html
    span_start_employees_pat = "".join(["^", "(", rel_qualifiers, space_pat, ")*",
                                        "(", employee_type_terms, space_pat, ")*",
                                        employee_terms])  # Employee terms at the beginning of a span

    emp_pat_list = [number_employees_pat, employed_num_pat, emp_type_emp_term_pat,
                    employed_end_span_pat, span_start_employees_pat,
                    workforce_terms]  # Workforce terms just need a broad net
    emp_pats = [re.compile(x, re.I) for x in emp_pat_list]

    emp_head_pre = "((([0-9a-z](([.][0-9a-z])|([0-9a-z][.]))*[0-9a-z.]?)|Full-Time|Our|Number of|Total) ?)?"
    emp_head_terms = "(Associates|Employees|Headcount|Personnel|Team Members|Staff|Workforce)"

    emp_head_raw = re.compile("r" + "".join(["[>]", emp_head_pre, emp_head_terms, "([.:])?[<]"]), re.I)
    emp_head = re.compile("".join(["^", emp_head_pre, emp_head_terms, "([.:])?$"]), re.I)

    results_dict = extract_paragraph_df(html_path_list, emp_pats, emp_head_raw, head_block_re)

    paragraph_input_df = results_dict['paragraphs']
    tbl_html_df = results_dict['tables']

    validation_id_bs = paragraph_input_df.acc_id.isin(val_accession_ids)
    paragraph_input_df.loc[validation_id_bs, 'split'] = 'val'

    tbl_html_df.loc[tbl_html_df.acc_id.isin(val_accession_ids),'split'] = 'val'

    paragraph_input_df.to_csv('data/paragraph_input_df_1.csv')
    tbl_html_df.to_csv('data/tbl_html_df_1.csv')

