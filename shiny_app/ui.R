ui <- fluidPage(theme = shinytheme("sandstone"),
  titlePanel("Extract information from unstructured text."),
    
  sidebarLayout(
    sidebarPanel(
      h5(tags$strong("Enter text to see what facts are extracted.")), 
      
      helpText("This demo currently uses terms related to employee counts."),
      
      textInput("text", label = h3("Text input"), value = init_text),
      
      hr(),

      actionButton("update", "Update View")
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      fluidRow(
       textOutput("entry_text")
      ),
      fluidRow(
        h3("Facts extracted from input:"),
        DT::dataTableOutput('facts')
      )
    )
  )
)