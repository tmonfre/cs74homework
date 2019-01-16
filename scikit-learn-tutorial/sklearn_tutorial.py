from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

gnb = GaussianNB()
gnb.fit(iris.data, iris.target)
y_pred = gnb.predict(iris.data)
y_probs = gnb.predict_proba(iris.data)

for i in range(y_pred.shape[0]):
    if y_pred[i] != iris.target[i]:
        print(y_probs[i])               # shows that when we are wrong, we aren't super confident in any one answer

size = iris.target.shape[0]
wrong_count = (iris.target != y_pred).sum()
percent_correct = (size - wrong_count) / size

print("NUMBER OF MISLABELED POINTS OUT OF %d POINTS: %d" % (iris.data.shape[0],wrong_count))
print("ACCURACY: " + str(percent_correct * 100) + "%")