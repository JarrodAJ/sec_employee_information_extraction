server <- function(input, output){
  
  #main <- import_main()
  
  
  
  
  # Return the requested dataset ----
  # Note that we use eventReactive() here, which depends on
  # input$update (the action button), so that the output is only
  # updated when the user clicks the button
  fact_data <- eventReactive(input$update, {
    fact_table(input$text) 
  }, ignoreNULL = TRUE, ignoreInit = FALSE)
  
  output$entry_text <- renderText({ 
    input$update
    isolate(input$text) })
  
  output$facts <- DT::renderDataTable(
    fact_data(), options = list(pageLength=25)
  )
  
  
}