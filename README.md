Spearmint-Sklearn
---------

Spearmint-Sklearn provides a sklearn interface to the well-known Spearmint package (https://github.com/JasperSnoek/spearmint).

To make it easy for use, we clone the source codes of spearmint-lite here and add a new wrapper function in "spearmint/search.py". For example, it can be used in the following way: 

```python
from sklearn import svm
from spearmint.search import GPEISklearn

sx = MinMaxScaler()
dt = datasets.load_breast_cancer()
x = sx.fit_transform(dt.data)
y = dt.target

ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
             'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

estimator = svm.SVC()
score_metric = make_scorer(accuracy_score, True)
cv = KFold(n_splits=5, random_state=0, shuffle=True)

clf = GPEISklearn(estimator, cv, ParaSpace, max_runs = 100, time_out = 10, refit = True, verbose = True)
clf.fit(x, y)
clf.logs
```
