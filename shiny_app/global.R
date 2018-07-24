library(tidyverse)
library(DT)
library(reticulate)
library(shinythemes) 
source_python("../relextract.py")
source_python("../emppipe.py")
init_text <- "As of February 23, 2017, we employed approximately 41,000 full-time Team Members and approximately 33,000 part-time Team Members."


fact_table <- function(text = init_text){
  fix_token_columns(
  add_units_and_values(
    make_fact_df(text, extract_emp_relations), 'quantity'), 
    c('subject', 'verb', 'quantity', 'word')
  ) %>% 
    select(data_values, subject, verb, quantity, word)
}
