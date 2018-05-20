# Shiny provides a family of functions that turn R objects into output for your user interface. Each function creates a specific type of output.
# 
# render function	creates
# renderDataTable	DataTable
# renderImage	images (saved as a link to a source file)
# renderPlot	plots
# renderPrint	any printed output
# renderTable	data frame, matrix, other table like structures
# renderText	character strings
# renderUI	a Shiny tag object or HTML

library(shiny)
library(GGally)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
   
  output$plot <- renderPlot({
    
    varGraph <- input$radio
    if(varGraph == 1){
        plot(swiss$Fertility, swiss$Infant.Mortality, data = swiss, col= swiss$Catholic)
    }
    else if(varGraph == 3){
      pairs(swiss, panel = panel.smooth, main = "swiss data",
                       col = 3 + (swiss$Catholic > 50))
      # varPlot <- varPlot + summary(lm(Fertility ~ . , data = swiss))
      # varPlot
    }
    else if(varGraph == 2){
      ggpairs(swiss)
    }
    
  })
 
})
