**DistanceClassifier** is a distance-based classifier based on the [scikit-learn](http://scikit-learn.org/) BaseEstimator class. It classifies instances based on how close they are to the distribution training examples for each class. It can use Mahalanobis distance or Euclidean distance. 

Usage
===

In Python: 

```python
from DistanceClassifier import DistanceClassifier
# initialize classifier 
dc = DistanceClassifier()
# train classifier
dc.fit(X,Y)
# make predictions on test set
y_test = dc.predict(X_test)
```

From the terminal:

```bash
python -m DistanceClassifier.DistanceClassifier path_to_data/data.csv
```

Acknowledgments
===
This method is being developed to study the genetic causes of human disease in the [Epistasis Lab at UPenn](http://epistasis.org). Work is partially supported by the [Warren Center for Network and Data Science](http://warrencenter.upenn.edu).  


