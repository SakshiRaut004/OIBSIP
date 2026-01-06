import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Iris.csv")
print(data.head())

X = data.iloc[:, 1:5].values   # measurements
y = data.iloc[:, 5].values    # species

print(X.shape)
print(y.shape)

sns.pairplot(
    data,
    hue="Species",
    diag_kind="kde"
)
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

sample = [[5.1, 3.5, 1.4, 0.2]]  # Example flower

prediction = model.predict(sample)
print("Predicted Species:", iris.target_names[prediction][0])
