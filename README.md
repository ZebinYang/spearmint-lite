Spearmint-Sklearn
---------

Spearmint-Sklearn is based on the well-known Spearmint package, which can be found in https://github.com/JasperSnoek/spearmint.
Here, we just provide a sklearn wrapper to make it easy to use. 

For example, it can be used as follows: 
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
