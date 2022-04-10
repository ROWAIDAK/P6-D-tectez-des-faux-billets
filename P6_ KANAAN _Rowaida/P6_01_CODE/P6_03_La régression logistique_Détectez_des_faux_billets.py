#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# In[22]:


df = pd.read_csv('datas/notes.csv' , header=0, sep=",", decimal=".")
df


# # Mission 3
# Modélisez les données à l'aide d'une régression logistique. Grâce à celle-ci, vous créerez un programme capable d'effectuer une prédiction sur un billet, c'est-à-dire de déterminer s'il s'agit d'un vrai ou d'un faux billet. Pour chaque billet, votre algorithme de classification devra donner la probabilité que le billet soit vrai. Si cette probabilité est supérieure ou égale à 0.5, le billet sera considéré comme vrai. Dans le cas contraire, il sera considéré comme faux.
# 
# 

# # La régression logistique
# La régression logistique est un modèle statistique permettant d’étudier les relations entre un ensemble de variables qualitatives Xi et une variable qualitative Y. Il s’agit d’un modèle linéaire généralisé utilisant une fonction logistique comme fonction de lien.
# 
# Un modèle de régression logistique permet aussi de prédire la probabilité qu’un événement arrive (valeur de 1) ou non (valeur de 0) à partir de l’optimisation des coefficients de régression. Ce résultat varie toujours entre 0 et 1. Lorsque la valeur prédite est supérieure à un seuil, l’événement est susceptible de se produire, alors que lorsque cette valeur est inférieure au même seuil, il ne l’est pas.

# ## Vérification de la colinéarité des variables
# Il est primordial de découvrir et quantifier à quel point deux variables sont liées. Ces relations peuvent être complexes et ne sont pas forcément visibles. Or certaines de ces dépendances affaiblissent les performances d’algorithme
# 
# Le facteur d'inflation de la variance (VIF) est une mesure de la quantité de multicolinéarité dans un ensemble de variables de Régression Logistique. Un VIF élevé indique que la variable indépendante associée est fortement colinéaire avec les autres variables du modèle. Nous pouvons avoir une idée des éventuels problèmes de colinéarité.

# In[23]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# In[24]:


reg_multi = smf.ols('diagonal  ~ height_left + height_right + margin_low + margin_up + length', data=df).fit()
variables = reg_multi.model.exog
[variance_inflation_factor(variables, i) for i in np.arange(1,variables.shape[1])]


# VIF < 10 il n'y aura pas d'influence liée à la colinéarité des variables. Si des variables colinéaires sont de facto fortement corrélées entre elles, deux variables corrélées ne sont pas forcément colinéaires. La régression logistique peut-être modélisée sur nos six variables explicatives, il ne semble pas avoir de frein possible, ni colinéarité, il n'y a non plus pas de valeur atypique influente.

# ## Modélisation des données à l'aide d'une Régression Logistique

# In[25]:


df['is_genuine'] = df['is_genuine'].apply(lambda x: 1 if x == True else 0)
df["is_genuine"] = df["is_genuine"].astype('category')
#set X and y
X = df.iloc[:, 1:7].values 
Y = df.iloc[:, 0].values


#split into train & test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


# Ici, l'ensemble de données est divisé en deux parties dans un rapport de 20/80. Cela signifie que 80% des données seront utilisées pour la formation des modèles et 20% pour les tests des modèles. Les données étant assez limitées, le choix des 20% pour les données de test est choisi.
# 
# 

# ## Évaluation du modèle à l'aide de la matrice de confusion.

# ### Model Fit

# In[26]:


#Instanciation d'un modèle nommé lr
lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[27]:


Y_pred_train= lr.predict(X_train)

confusion_matrix1 = confusion_matrix(Y_train, Y_pred_train)
print(confusion_matrix1) 


# In[28]:



#Représentation graphique de la Matrice de confusion
sns.heatmap(pd.DataFrame(confusion_matrix1), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig("graphiques/09_1 confusion_matrix.png")
plt.show()

print(metrics.classification_report(Y_train, Y_pred_train))


# ### Model Predict

# In[29]:


Y_pred = lr.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix) 
#Représentation graphique de la Matrice de confusion
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig("graphiques/09_2 confusion_matrix.png")
plt.show()

#Créez un rapport texte montrant les principales métriques de classification.
print ("classification_report")
print ('------------------------')
print(metrics.classification_report(Y_test, Y_pred))


# In[31]:


print ("accuracy", metrics.accuracy_score(Y_test, Y_pred))
#Score de classification de précision. التنبؤات الصحيحة من كل التنبؤات#


# LogisticRegression veut savoir à 97 % si les pièces sont vraies ou fausses
# 

# In[42]:


print ("precision score",metrics.precision_score(Y_test, Y_pred, average='macro')) 
#La précision est le rapport tp / (tp + fp) où tp est le nombre de vrais positifs et fp le nombre de faux positifs.


# In[44]:


#Mesure AUC (Area Under the Curve): Aire sous la courbe
#from sklearn import metrics
metrics.auc(false_positive_rate, true_positive_rate)  


# # Courbe ROC et indicateur AUC 
# 

# In[35]:


#Courbe ROC
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs ')
plt.title('Courbe ROC')



plt.show()


# In[36]:


print ("AUC&ROC",metrics.roc_auc_score(Y_test, Y_pred))


# #### Rappel :
# 
# dans le pire des cas, AUC = 0.5\
# dans le meilleur des cas, AUC = 1\
# Ici, nous avons un excellent modèle

# L’idée de **la courbe ROC** est de faire varier le «seuil» de 1 à 0 et, pour chaque cas, calculer le taux de vrai positif et de faux positif que l’on reporte dans un graphique: en abscisse le TFP, en ordonnée le TVP.
# 
# Facile à interpréter, ne nécessite pas de mise à l'échelle, ni de calcul complexe. La régression logistique fournit un score de probabilité pour les observations. Dans ce cas précis, nous sommes très proche du classifieur optimal.

# In[37]:


metrics.homogeneity_score(Y_test, Y_pred)
#Métrique d'homogénéité d'un étiquetage de cluster compte tenu d'une vérité terrain


# In[38]:


#Métrique d'exhaustivité d'un étiquetage de cluster compte tenu d'une vérité terrain.
metrics.completeness_score(Y_test, Y_pred)


# L’idée de la courbe ROC est de faire varier le «seuil» de 1 à 0 et, pour chaque cas, calculer le taux de vrai positif et de faux positif que l’on reporte dans un graphique: en abscisse le TFP, en ordonnée le TVP.
# 
# Facile à interpréter, ne nécessite pas de mise à l'échelle, ni de calcul complexe. La régression logistique fournit un score de probabilité pour les observations. Dans ce cas précis, nous sommes en du classifieur optimal.

# ## L'évaluation du modèle peut aussi se faire par la courbe ROC et sa métrique AUC.

# ### Application sur le fichier test_exemple.csv

# In[39]:


#Prédiction faite à partir du fichier "test_example.csv"
df_example = pd.read_csv('datas/test_example.csv')
df_example


# In[40]:


#Préparation des données
X = df_example.copy()
X = X.iloc[:, :-1]
#Utilisation du modèle de prédiction 'lr'
probability = lr.predict_proba(X.values)[:, 1]
#Probabilités des billets établies 
proba = pd.Series(probability.round(3), name='value')
#Intégration des probabilités dans le jeu de données
df_example_final = pd.concat([df_example, proba], axis=1)
df_example_final


# L'algorithme de classification donnera la probabilité que le billet soit vrai. Si cette probabilité est supérieure ou égale à 0.5, le billet sera considéré comme vrai. Dans le cas contraire, il sera considéré comme faux.

# In[41]:


#Résultats de la classification prédictive :
resultat = []
for i in df_example_final['value'] >= .5:
    if i is True :
        resultat.append('Vrai Billet')
    else :
        resultat.append('Faux Billet')

df_example_final['resultat'] = resultat
df_example_final


# In[ ]:





# In[ ]:





# In[ ]:




