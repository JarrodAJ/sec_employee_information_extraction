ui <- navbarPage("Information Extraction", theme = shinytheme("sandstone"),
  tabPanel("Project Summary", 
   h2("Motivation"), 
   p("There are enormous amounts of data available in text, but it’s very hard to get information from most of it. Getting actual numbers from documents (web pages, SEC filings, EMRs, etc.) is very resource intensive."),
   p("For example, in this project, I am trying to find out how many employees every Fortune 500 company has, and I’m trying to distinguish between part-time employees and full-time employees. Sometimes it’s easy to find and verify:"),
   p("\"We employed 2,000 employees.\""),
   p("One could imagine just searching for \"employees\" in a document, or using programming to find a pattern of a number followed by \"employees.\" However, one also has to search for other terms, and the patterns break easily, as in these examples:"),
    p("The number of regular employees was 71.1 thousand, 73.5 thousand, and 75.3 thousand at years ended 2016, 2015 and 2014, respectively."),
     p("\"Total workforce level at December 31, 2016 was approximately 150,500.\""),
     p("Cases such as these require an approach that can handle different vocabularies, different years, etc. ")
   ),
  tabPanel("Interactive demo",
    titlePanel("Extract employee count information from unstructured text"),
      
    sidebarLayout(
      sidebarPanel(
        h5(tags$strong("Enter text to see what facts are extracted")), 
        
        helpText("This demo currently uses terms related to employee counts."),
        
        textInput("text", label = h3("Text input"), value = init_text),
        
        hr(),
  
        actionButton("update", "Extract Facts")
      ),
      
      # Main panel for displaying outputs ----
      mainPanel(
        fluidRow(
          tags$h3("Text entered:"),
          textOutput("entry_text")
        ),
        fluidRow(
          h3("Facts extracted from input:"),
          DT::dataTableOutput('facts')
        )
      )
    )
  )
)