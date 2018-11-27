import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
from sklearn.metrics import accuracy_score
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # for plot styling
import warnings

data = pd.read_csv('features/merged/merged_supervised.csv', sep='§')
data.drop('Unnamed: 0',axis= 1, inplace=True)


y_train = pd.read_csv('data/external/y_train_violent.csv', sep='§')
y_test = pd.read_csv('data/external/y_test_violent.csv', sep='§')
y_train.drop('Unnamed: 0',axis= 1, inplace=True)
y_test.drop('Unnamed: 0',axis= 1, inplace=True)

train = pd.merge(data,y_train, on= 'Sequence')

test = pd.merge(data,y_test, on= 'Sequence')

X_train = train.drop(['Sequence','Violent'],axis= 1)
X_test = test.drop(['Sequence','Violent'],axis= 1)
y_test = test['Violent']
y_train = train['Violent']
model = RandomForestClassifier(n_estimators=20,random_state=42)
model.fit(X_train,y_train)


feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Random Forest feature importance for violent content classification')
plt.tight_layout()
plt.show()

print(accuracy_score(model.predict(X_test),y_test))

data = pd.read_csv('features/merged/merged_supervised.csv', sep='§')
data.drop('Unnamed: 0',axis= 1, inplace=True)


y_train = pd.read_csv('data/external/y_train_exterieur.csv', sep='§')
y_test = pd.read_csv('data/external/y_test_exterieur.csv', sep='§')
y_train.drop('Unnamed: 0',axis= 1, inplace=True)
y_test.drop('Unnamed: 0',axis= 1, inplace=True)

train = pd.merge(data,y_train, on= 'Sequence')

test = pd.merge(data,y_test, on= 'Sequence')

X_train = train.drop(['Sequence','Exterieur'],axis= 1)
X_test = test.drop(['Sequence','Exterieur'],axis= 1)
y_test = test['Exterieur']
y_train = train['Exterieur']
model = RandomForestClassifier(n_estimators=20,random_state=42)
model.fit(X_train,y_train)

feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Features (avg over folds)')
plt.tight_layout()
plt.show()

print(accuracy_score(model.predict(X_test),y_test))