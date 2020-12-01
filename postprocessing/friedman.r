# Originally published on:
# http://www.r-statistics.com/2010/02/post-hoc-analysis-for-friedmans-test-r-code

# good read: 	http://en.wikipedia.org/wiki/Friedman_test
# 				http://finzi.psych.upenn.edu/R/library/coin/html/SymmetryTests.html
# could have used: block = with(data, eval(parse(text = all.vars(formu)[3])))

friedman.test.with.post.hoc <- function(formu, data, to.print.friedman = T, to.post.hoc.if.signif = T,  to.plot.parallel = T, to.plot.boxplot = T, signif.P = .05, color.blocks.in.cor.plot = T, jitter.Y.in.cor.plot =F)
{
  # formu is a formula of the shape: 	Y ~ X | block
  # data is a long data.frame with three columns:    [[ Y (numeric), X (factor), block (factor) ]]
  
  # Note: This function doesn't handle NA's! In case of NA in Y in one of the blocks, then that entire block should be removed.
  
  
  # Loading needed packages
  if(!require(coin))
  {
    print("You are missing the package 'coin', we will now try to install it...")
    install.packages("coin")		
    library(coin)
  }
  
  if(!require(multcomp))
  {
    print("You are missing the package 'multcomp', we will now try to install it...")
    install.packages("multcomp")
    library(multcomp)
  }
  
  if(!require(colorspace))
  {
    print("You are missing the package 'colorspace', we will now try to install it...")
    install.packages("colorspace")
    library(colorspace)
  }
  
  
  # get the names out of the formula
  formu.names <- all.vars(formu)
  Y.name <- formu.names[1]
  X.name <- formu.names[2]
  block.name <- formu.names[3]
  
  if(dim(data)[2] >3) data <- data[,c(Y.name,X.name,block.name)]	# In case we have a "data" data frame with more then the three columns we need. This code will clean it from them...
  
  # Note: the function doesn't handle NA's. In case of NA in one of the block T outcomes, that entire block should be removed.
  
  # stopping in case there is NA in the Y vector
  if(sum(is.na(data[,Y.name])) > 0) stop("Function stopped: This function doesn't handle NA's. In case of NA in Y in one of the blocks, then that entire block should be removed.")
  
  # make sure that the number of factors goes with the actual values present in the data:
  data[,X.name ] <- factor(data[,X.name ])
  data[,block.name ] <- factor(data[,block.name ])
  number.of.X.levels <- length(levels(data[,X.name ]))
  if(number.of.X.levels == 2) { warning(paste("'",X.name,"'", "has only two levels. Consider using paired wilcox.test instead of friedman test"))}
  
  # making the object that will hold the friedman test and the other.
  the.sym.test <- symmetry_test(formu, data = data,	### all pairwise comparisons	
                                teststat = "max",
                                xtrafo = function(Y.data) { trafo( Y.data, factor_trafo = function(x) { model.matrix(~ x - 1) %*% t(contrMat(table(x), "Tukey")) } ) },
                                ytrafo = function(Y.data){ trafo(Y.data, numeric_trafo = rank, block = data[,block.name] ) }
  )
  # if(to.print.friedman) { print(the.sym.test) }
  
  
  if(to.post.hoc.if.signif)
  {
    if(pvalue(the.sym.test) < signif.P)
    {
      # the post hoc test
      The.post.hoc.P.values <- pvalue(the.sym.test, method = "single-step")	# this is the post hoc of the friedman test
      
      
      # plotting
      if(to.plot.parallel & to.plot.boxplot)	par(mfrow = c(1,2)) # if we are plotting two plots, let's make sure we'll be able to see both
      
      if(to.plot.parallel)
      {
        X.names <- levels(data[, X.name])
        X.for.plot <- seq_along(X.names)
        plot.xlim <- c(.7 , length(X.for.plot)+.3)	# adding some spacing from both sides of the plot
        
        if(color.blocks.in.cor.plot) 
        {
          blocks.col <- rainbow_hcl(length(levels(data[,block.name])))
        } else {
          blocks.col <- 1 # black
        }					
        
        data2 <- data
        if(jitter.Y.in.cor.plot) {
          data2[,Y.name] <- jitter(data2[,Y.name])
          par.cor.plot.text <- "Parallel coordinates plot (with Jitter)"				
        } else {
          par.cor.plot.text <- "Parallel coordinates plot"
        }				
        
        # adding a Parallel coordinates plot
        matplot(as.matrix(reshape(data2,  idvar=X.name, timevar=block.name,
                                  direction="wide")[,-1])  , 
                type = "l",  lty = 1, axes = FALSE, ylab = Y.name, 
                xlim = plot.xlim,
                col = blocks.col,
                main = par.cor.plot.text)
        axis(1, at = X.for.plot , labels = X.names) # plot X axis
        axis(2) # plot Y axis
        points(tapply(data[,Y.name], data[,X.name], median) ~ X.for.plot, col = "red",pch = 4, cex = 2, lwd = 5)
      }
      
      if(to.plot.boxplot)
      {
        # first we create a function to create a new Y, by substracting different combinations of X levels from each other.
        subtract.a.from.b <- function(a.b , the.data)
        {
          the.data[,a.b[2]] - the.data[,a.b[1]]
        }
        
        temp.wide <- reshape(data,  idvar=X.name, timevar=block.name,
                             direction="wide") 	#[,-1]
        wide.data <- as.matrix(t(temp.wide[,-1]))
        colnames(wide.data) <- temp.wide[,1]
        
        Y.b.minus.a.combos <- apply(with(data,combn(levels(data[,X.name]), 2)), 2, subtract.a.from.b, the.data =wide.data)
        names.b.minus.a.combos <- apply(with(data,combn(levels(data[,X.name]), 2)), 2, function(a.b) {paste(a.b[2],a.b[1],sep=" - ")})
        
        the.ylim <- range(Y.b.minus.a.combos)
        the.ylim[2] <- the.ylim[2] + 2*max(sd(Y.b.minus.a.combos))+1	# adding some space for the labels
        is.signif.color <- ifelse(The.post.hoc.P.values < .05 , "green", "grey")
        
        op <- par(mar = c(2,13,2,2) + 0.1)
        boxplot(Y.b.minus.a.combos,
                names = names.b.minus.a.combos ,
                col = is.signif.color,
                main = "Boxplots of Rank Differences",
                ylim = the.ylim,
                horizontal=TRUE,
                las=1
        )
        par(op)
        legend("topright", legend = paste(rev(names.b.minus.a.combos), rep(" ; PostHoc P.value:", number.of.X.levels),round(rev(The.post.hoc.P.values),5)) , fill =  rev(is.signif.color) )
        abline(v = 0, col = "blue")
        
      }
      
      list.to.return <- list(Friedman.Test = the.sym.test, PostHoc.Test = The.post.hoc.P.values)
      if(to.print.friedman) {print(list.to.return)}				
      return(list.to.return)
      
    }	else {
      print("The results were not significant, There is no need for a post hoc test")
      return(the.sym.test)
    }					
  }
  
  # Original credit (for linking online, to the package that performs the post hoc test) goes to "David Winsemius", see:
  # http://tolstoy.newcastle.edu.au/R/e8/help/09/10/1416.html
}



# friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting)	# the same with our function. With post hoc, and cool plots
# boxplot(Taste ~ Wine  ,WineTasting)


# example

friedman.test.with.post.hoc.example <- function()
{
  
  
  # Loading needed packages
  if(!require(coin))
  {
    install.packages("coin")
    library(coin)
  }
  
  if(!require(multcomp))
  {
    install.packages("multcomp")
    library(multcomp)
  }
  
  
  
  
  ### Comparison of three Wine ("Wine A", "Wine B", and
  ###  "Wine C") for rounding first base. 
  WineTasting <- data.frame(
    Taste = c(5.40, 5.50, 5.55,
              5.85, 5.70, 5.75,
              5.20, 5.60, 5.50,
              5.55, 5.50, 5.40,
              5.90, 5.85, 5.70,
              5.45, 5.55, 5.60,
              5.40, 5.40, 5.35,
              5.45, 5.50, 5.35,
              5.25, 5.15, 5.00,
              5.85, 5.80, 5.70,
              5.25, 5.20, 5.10,
              5.65, 5.55, 5.45,
              5.60, 5.35, 5.45,
              5.05, 5.00, 4.95,
              5.50, 5.50, 5.40,
              5.45, 5.55, 5.50,
              5.55, 5.55, 5.35,
              5.45, 5.50, 5.55,
              5.50, 5.45, 5.25,
              5.65, 5.60, 5.40,
              5.70, 5.65, 5.55,
              6.30, 6.30, 6.25),
    Wine = factor(rep(c("Wine A", "Wine B", "Wine C"), 22)),
    Taster = factor(rep(1:22, rep(3, 22))))
  
  with(WineTasting , boxplot( Taste  ~ Wine )) # boxploting 
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting)	# the same with our function. With post hoc, and cool plots
  
  
  ### classical global test
  friedman_test(Taste ~ Wine | Taster, data = WineTasting)
  
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting[1:15,])	# showing what happens when results are not signif
  
  #	fo <- Taste ~ Wine | Taster
  #  mtrace(temp)
  # mtrace.off()
  
  
  # what happens in the case of only two levels in X
  WineTasting2 <- WineTasting[WineTasting[,2] %in% levels(WineTasting[,2])[2:3],]
  WineTasting2[,2] <- factor(WineTasting2[,2] )
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting2)
  
  # what happens in case of an NA
  WineTasting2 <- WineTasting
  WineTasting2[1,1] <- NA
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting2)
}							 

#friedman.test.with.post.hoc.example()













friedman.test.with.post.hoc.example2 <- function()
{
  
  
  # Loading needed packages
  if(!require(coin))
  {
    install.packages("coin")
    library(coin)
  }
  
  if(!require(multcomp))
  {
    install.packages("multcomp")
    library(multcomp)
  }
  
  
  
  
  ### Comparison of three Wine ("Wine A", "Wine B", and
  ###  "Wine C") for rounding first base. 
  WineTasting <- data.frame(
    Taste = c(5.40, 5.50, 5.55,
              5.85, 5.70, 5.75,
              5.20, 5.60, 5.50,
              5.55, 5.50, 5.40,
              5.90, 5.85, 5.70,
              5.45, 5.55, 5.60,
              5.40, 5.40, 5.35,
              5.45, 5.50, 5.35,
              5.25, 5.15, 5.00,
              5.85, 5.80, 5.70,
              5.25, 5.20, 5.10,
              5.65, 5.55, 5.45,
              5.60, 5.35, 5.45,
              5.05, 5.00, 4.95,
              5.50, 5.50, 5.40,
              5.45, 5.55, 5.50,
              5.55, 5.55, 5.35,
              5.45, 5.50, 5.55,
              5.50, 5.45, 5.25,
              5.65, 5.60, 5.40,
              5.70, 5.65, 5.55,
              6.30, 6.30, 6.25),
    Wine = factor(rep(c("Wine A", "Wine B", "Wine C"), 22)),
    Taster = factor(rep(1:22, rep(3, 22))))
  
  ### classical global test
  friedman_test(Taste ~ Wine | Taster, data = WineTasting)
  
  #	fo <- Taste ~ Wine | Taster
  
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting)	# the same with our function. With post hoc, and cool plots
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting[1:15,])	# showing what happens when results are not signif
  
  #  mtrace(temp)
  # mtrace.off()
  
  
  # what happens in the case of only two levels in X
  WineTasting2 <- WineTasting[WineTasting[,2] %in% levels(WineTasting[,2])[2:3],]
  WineTasting2[,2] <- factor(WineTasting2[,2] )
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting2)
  
  # what happens in case of an NA
  WineTasting2 <- WineTasting
  WineTasting2[1,1] <- NA
  friedman.test.with.post.hoc(Taste ~ Wine | Taster ,WineTasting2)
}							 

#friedman.test.with.post.hoc.example2()