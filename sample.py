#!/home/kfujii/anaconda3/bin/python

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Irisデータセットをロード
iris = load_iris()
X, y = iris.data, iris.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.3, random_state=123)
# 決定木をインスタンス化
clf = DecisionTreeClassifier(random_state=123)
param_grid = {'max_depth': [3, 4, 5]}

# 10分割の交差検証を実行
cv = GridSearchCV(clf, param_grid=param_grid, cv=10)
cv.fit(X_train, y_train)

# 最適な深さを確認する
print(cv.best_params_)

# 最適なモデルを確認する
print(cv.best_estimator_)


# テストデータのクラスラベルを予測する
y_pred = cv.predict(X_test)
print(y_pred)
