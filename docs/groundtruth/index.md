<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<!-- <script src="vega-embed-6.15.0.min.js"></script> -->
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>


<script type="text/javascript">
    var view;

    fetch('../plots/srGT.json')
      .then(res => res.json())
      .then(spec => render(spec, "#srGT"))
      .catch(err => console.error(err));
    fetch('../plots/accGT.json')
      .then(res => res.json())
      .then(spec => render(spec, "#accGT"))
      .catch(err => console.error(err));

    function render(spec, cont) {
      view = new vega.View(vega.parse(spec), {
        renderer:  'canvas',  // renderer (canvas or svg)
        container: cont,   // parent DOM container
        hover:     true       // enable hover processing
      });
      return view.runAsync();
    }
  </script>

# Results for Ground-truth Problems

## Symbolically-verfied Solutions

How often a method finds a model symbolically equivalent to the ground-truth process

<div id="srGT"></div>

## Accuracy Solutions

How often a method finds a model with test set `R^2>0.999`

<div id="accGT"></div>

<br><br>
