# Employee Count Extraction
This repo contains code to extract and classify employee counts from unstructured text in html SEC filings (financial reports, such as [Form 10-K](https://en.wikipedia.org/wiki/Form_10-K), that many companies must file); the final output is a table. A set of "golden," labeled data was provided for evaluation purposes. 

Currently, the code correctly extracts and labels 87% of the facts that exist in the golden dataset (for the validation subest - the test subset has not been evaluated yet). 

## Motivation

There are enormous amounts of data available in text, but it’s very hard to get information from most of it. Getting actual numbers from documents (web pages, SEC filings, EMRs, etc.) is very resource intensive. I hope to use this project to build a fact extraction framework (specifically for extracting quantities) that I can use in other domains. 

SEC filings are notoriously inconsistent in sentence structure, word usage, etc. For example, the first sentence below is easy to parse, but the next two require additional handling. 

### Example inputs and outputs:
(Note that this project actually starts from html verions of 50-200+ page documents)  
__Input:__

1. [filing](https://www.sec.gov/Archives/edgar/data/14846/000001484616000062/brtrealty10-k2016.htm) 
`"(Including our full and part-time personnel , we estimate that we have the equivalent of 12 full time employees." `
2. [filing link](https://www.sec.gov/Archives/edgar/data/34088/000003408817000017/xom10k2016.htm) 
`"The number of regular employees was 71.1 thousand, 73.5 thousand, and 75.3 thousand at years ended 2016, 2015 and 2014, respectively. Regular employees are defined as active executive, management, professional, technical and wage employees who work full time or part time for the Corporation and are covered by the Corporation’s benefit plans and programs. Regular employees do not include employees of the company‑operated retail sites (CORS). The number of CORS employees was 1.6 thousand, 2.1 thousand, and 8.4 thousand at years ended 2016, 2015 and 2014, respectively. The decrease in CORS employees reflects the multi‑year transition of the company‑operated retail network to a more capital‑efficient Branded Wholesaler model."`

3. [filing link](https://www.sec.gov/Archives/edgar/data/12927/000001292717000006/a201612dec3110k.htm)
`"Total workforce level at December 31, 2016 was approximately 150,500."`


__Output:__

doc_id|data values|quantity_type|subject|verb|quantity|type_token|word
--|-------|--------------|--------------------|--------|--------------|---------|---------------
1|12|Full-Time Employees|we|have|12|full time|employees
2|71100|Other Employees|The number of regular employees|was|71.1 thousand|regular|employees
2|1600|Other Employees|The number of CORS employees|was|1.6 thousand|CORS|employees
3|150500|Other Employees|level|was|150,500||workforce


## General process
The html files are first parsed into potentially-relevant chunks. The chunks (usually paragraphs) are then passed to a natural-language processing pipeline. The pipeline inspects each sentence for "cues" that the sentence contains information about employeee counts. Finally, the code extracts and cleans "facts" about employee counts, and produces a table that is ready for database ingestion. 
