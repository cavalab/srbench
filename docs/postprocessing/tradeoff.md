<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<!-- <script src="vega-embed-6.15.0.min.js"></script> -->
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>


# Benchmarking Results

This page summarizes the results of the postprocessing notebooks found in this folder. 

Results are summarized over datasets. 


## Accuracy-Complexity-Time Trade-offs

Considering the accuracy and simplicity of models simultaneously, this figure illustrates the trade-offs made by each method. 
Methods lower and to the left produce models with better trade-offs between accuracy and simplicity. 

<div id="paretoR2Size"></div>

<br><br>

<div id="paretoR2Time"></div>

<br><br>

<div id="paretoTimeSize"></div>

<br><br>

<script type="text/javascript">
  vegaEmbed('#paretoR2Size', "../plots/paretoR2Size.json").then(function(result) {
    // Access the Vega view instance (https://vega.github.io/vega/docs/api/view/) as result.view
  }).catch(console.error);
  vegaEmbed('#paretoR2Time', "../plots/paretoR2Time.json").then(function(result) {
    // Access the Vega view instance (https://vega.github.io/vega/docs/api/view/) as result.view
  }).catch(console.error);
  vegaEmbed('#paretoTimeSize', "../plots/paretoTimeSize.json").then(function(result) {
    // Access the Vega view instance (https://vega.github.io/vega/docs/api/view/) as result.view
  }).catch(console.error);
  
  </script>
