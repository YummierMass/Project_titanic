import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore')
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Wczytanie danych zbioru train oraz test do zmiennych:
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print('Liczba próbek w zbiorze train wynosi: {}.'.format(train_df.shape[0]))

# preview test data
test_df.head()
print('Liczba próbek w zbiorze test wynosi: {}.'.format(test_df.shape[0]))
# sprawdzenie brakujących wartości w zbiorze train
train_df.isnull().sum()

# wyświetlenie procentu brakujących wartości kolumny age
print('Procent brakujących wartości dla kolumny Age wynosi: %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))

ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

print('Średnia dla wieku wynosi: %.2f' %(train_df["Age"].mean(skipna=True)))
print('Mediana dla wieku wynosi: %.2f' %(train_df["Age"].median(skipna=True)))

# wyświetlenie procentu brakujących wartości kolumny Cabin
print('Procent brakujących wartości dla kolumny Cabin wynosi: %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))

# wyświetlenie procentu brakujących wartości kolumny Embarked
print('Procent brakujących wartości dla kolumny Embarked wynosi: %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.shape[0])*100))

print('Liczba pasażerów pogrupowanych według portu, w którym wsiedli na statek (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(train_df['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train_df, palette='Set2')
plt.show()

train_data = train_df.copy() # kopiowanie zbioru 
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True) # zastąpienie brakujących rekordów kolumny Age wartością 28
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True) # zastąpienie brakujących rekordów kolumny Embarked wartością S
train_data.drop('Cabin', axis=1, inplace=True) # usunięcie kolumny Cabin

# sprawdzenie, czy po dokonanych zmianach istnieją brakujące wartości w kolumnach
train_data.isnull().sum()

# podgląd dostosowanych zmian
train_data.head()

plt.figure(figsize=(15,8))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Niezmodyfikowane wartości wieku', 'Dodany wiek w postaci wartości mediany'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

## Utworzenie zmiennych pomocniczych do podróżowania samemu
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)

# create categorical variables and drop some variables
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()

test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()

plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Ocalali', 'Zmarli'])
plt.title('Wykres gęstości wieku dla populacji, która przeżyła i populacji zmarłej')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

plt.figure(figsize=(20,8))
avg_survival_byage = final_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
plt.show()

final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)

plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Ocalali', 'Zmarli'])
plt.title('Wykres gęstości dla kolumny Fare w przypadku ocalałej oraz zmarłej populacji')
ax.set(xlabel='Fare')
plt.xlim(-20,200)
plt.show()

sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")
plt.show()

sns.barplot('Embarked', 'Survived', data=train_df, color="teal")
plt.show()

sns.barplot('TravelAlone', 'Survived', data=final_train, color="mediumturquoise")
plt.show()

sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X = final_train[cols]
y = final_train['Survived']
# Budowanie logreg oraz obliczanie wagi feature
model = LogisticRegression()
# utworzenie modelu RFE oraz wybranie 8 atrybutów
rfe = RFE(model, 8)
rfe = rfe.fit(X, y)
# podsumowanie wybrania atrybutów
print('Wybrane cechy: %s' % list(X.columns[rfe.support_]))

from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optymalna liczba wybranych cech: %d" % rfecv.n_features_)
print('Wybrane cechy: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Liczba wybranych cech")
plt.ylabel("Wynik walidacji krzyżowej (liczba poprawnych klasyfikacji)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
                     'Embarked_S', 'Sex_male', 'IsMinor']
X = final_train[Selected_features]

plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# utworzenie zmiennej X (cechy) i y (odpowiedzi)
X = final_train[Selected_features]
y = final_train['Survived']

# wykorzystanie train/test z losowymi wartościami
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# sprawdzenie wyniku klasyfikacji regresji logistycznej
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Wyniku podziału Train/Test:')
print(logreg.__class__.__name__+" dokładność wynosi: %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" utrata logarytmiczna wynosi: %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc wynosi %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # indeks pierwszego progu, dla którego czułość jest większa niż 0,95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='krzywa ROC (obszar = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Wskaźnik wyników fałszywie dodatnich (1 - swoistość)', fontsize=14)
plt.ylabel('Prawdziwa pozytywna stawka (przywołanie)', fontsize=14)
plt.title('Krzywa charakterystyki pracy odbiornika (ROC)')
plt.legend(loc="lower right")
plt.show()

print("Stosując próg %.3f " % thr[idx] + "gwarantuje wrażliwość %.3f " % tpr[idx] +  
      "i specyfikę %.3f" % (1-fpr[idx]) + 
      ", wliczając fałszywie dodatni wskaźnik %.2f%%." % (np.array(fpr[idx])*100))

# 10-krotna regresja logistyczna z walidacją krzyżową
logreg = LogisticRegression()
# Wykorzystanie funkcji cross_val_score
# Przekazuję całość X i y, a nie X_train czy y_train, to zajmuje się dzieleniem danych
# cv=10 dla 10 kroków
# rezultaty = {'accuracy', 'neg_log_loss', 'roc_auc'} dla miernika oceny - choć jest ich wiele
scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
print('K-krotne wyniki walidacji krzyżowej:')
print(logreg.__class__.__name__+" średnia dokładność to %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" średnia log_loss to %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" średnia auc to %2.3f" % scores_auc.mean())

from sklearn.model_selection import cross_validate

scoring = {'dokładność': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-krotne wyniki walidacji krzyżowej:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" średnia %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))

from sklearn.model_selection import GridSearchCV

X = final_train[Selected_features]

param_grid = {'C': np.arange(1e-05, 3, 0.1)}
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}

gs = GridSearchCV(LogisticRegression(), return_train_score=True,
                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')

gs.fit(X, y)
results = gs.cv_results_

print('='*20)
print("Najlepsze parametry: " + str(gs.best_estimator_))
print("Najlepsze parametry: " + str(gs.best_params_))
print('Najlepszy wynik:', gs.best_score_)
print('='*20)

plt.figure(figsize=(10, 10))
plt.title("## GridSearchCV - ocenianie przy jednoczesnym użyciu wielu punktów ",fontsize=16)

plt.xlabel("Odwrotność siły regularyzacji: C")
plt.ylabel("Wynik")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, param_grid['C'].max()) 
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
        
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()