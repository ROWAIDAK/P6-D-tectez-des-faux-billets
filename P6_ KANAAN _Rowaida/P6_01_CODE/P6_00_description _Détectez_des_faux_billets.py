#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


# In[19]:


#pip install pingouin


# # Scénario
# Votre société de consulting informatique vous propose une nouvelle mission au ministère de l'Intérieur, dans le cadre de la lutte contre la criminalité organisée, à l'Office central pour la répression du faux monnayage. Votre mission  : créer un algorithme de détection de faux billets.
# 
# Vous vous voyez déjà en grand justicier combattant sans relâche la criminalité organisée en pianotant à mains de maître votre ordinateur, pour façonner ce fabuleux algorithme  qui traquera la moindre fraude et permettra de mettre à jour les réseaux secrets de faux-monnayeurs ! La classe, non ?
# 
# ... Bon, si on retombait les pieds sur terre? Travailler pour la police judiciaire, c'est bien, mais vous allez devoir faire appel à vos connaissances en statistiques, alors on y va !
# 
# # Les données
# La PJ vous transmet un jeu de données contenant les caractéristiques géométriques de billets de banque. Pour chacun d'eux, nous connaissons :
# 
#    1. la longueur du billet (en mm) ;
#    2. la hauteur du billet (mesurée sur le côté gauche, en mm) ;
#    3. La hauteur du billet (mesurée sur le côté droit, en mm) ;
#    4. la marge entre le bord supérieur du billet et l'image de celui-ci (en mm) ;
#    5. la marge entre le bord inférieur du billet et l'image de celui-ci (en mm) ;
#    6. la diagonale du billet (en mm).
#    
# # Votre mission
# 
# 
# # Mission 0
# Afin d'introduire votre analyse, effectuez une brève description des données (analyses univariées et bivariées).

# In[20]:


df = pd.read_csv('datas/notes.csv' , header=0, sep=",", decimal=".")
df


# ## Analyse Univariée

# In[21]:


df.shape


# In[22]:


df.describe(include = 'all')


# In[23]:


df.info()


# In[24]:


df.isnull().sum()


# In[25]:


print(df.duplicated().sum())


# In[26]:


data = df.groupby("is_genuine").count()
data


# In[27]:


data.plot.pie(y="diagonal",figsize=(8, 8),
                                explode = [0, 0.1],
                                labels = ["Faux billets", "Vrais billets"],
                                autopct = '%1.1f%%',
                                pctdistance = 0.3, labeldistance = 0.5)

plt.title('Répartition des vrais et faux billets du jeu de données ', 
  loc='center', 
  fontsize=22)
plt.savefig('graphiques/01. DESCRIBE_repartition_vrai_faux.png')
plt.show()


# ### Présentation des données
# Le fichier notes.csv liste les caractéristiques de 170 billets de banque différents. On dispose de 6 mesures toutes exprimées en millimètres : diagonale, hauteurs à gauche et à droite, marges hautes et basses, longueur.  ainsi qu'une colonne de type booléenne permettant d'authentifier le billet (vrai ou faux billet de banque).
# 
# Il n'y a aucune valeur manquante ou aberrante, aucun nettoyage n'est donc requis.
# 
# 41,2 % des billets présents dans le jeu de données sont des faux. Analysons les variables indépendamment pour vérifier leur distribution globale.
# 

# ### Le calcul de la distribution empirique est la première étape pour la représentation graphique d'une variable qualitative.
# 

# In[28]:


for column in df[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']] :
    plt.figure(figsize = (8,5))
    sns.histplot(x=column, data=df, kde=True, color='#2cb2ff')
    plt.savefig("graphiques/02. Histogramme " + str(column) +".jpg", dpi=500, bbox_inches='tight', pad_inches=0.5)

    plt.xlabel(column)


# In[29]:


#normality of variables in df
import pingouin as pg
pg.normality(df, method='shapiro', alpha=0.05).drop('is_genuine')

#normality: test de normalité univarié.


# Pour les variables margin_low et length, le niveau de test 5% ne permet pas de conclure à l'adéquation à la loi normale, ce que semblent aussi montrer les histogrammes

# ## Analyse Bivariée

# In[30]:


for column in df[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']] : 
        plt.figure(figsize = (8,5))
        sns.boxplot( x=column, y='is_genuine', data=df, orient='h', palette=('#4cb2ff', '#61ba86'))
        plt.savefig("graphiques/03. Boxplot " + str(column) +".jpg", dpi=500, bbox_inches='tight', pad_inches=0.5)


# On remarque que les variables où il existe une différence un peu plus marquée entre les vrais et faux billets sont les variables length et margin_low 
# 
# Les faux billets sont plus courts que les vrais, et leur marge basse est à l'inverse nettement plus longue.

# In[31]:


g = sns.pairplot(df, hue='is_genuine', markers=['o','s'], corner=True)
g.map_lower(sns.kdeplot, levels=2, color='.2')
plt.savefig("graphiques/04. Pairplot.jpg", dpi=500, bbox_inches='tight', pad_inches=0.5)
plt.show()


# sns.pairplot trace l'histogramme de chaque variable quantitative, pour chaque classe de la variable catégorielle is_genuine.
# 
# On retrouve ces corrélations sur les graphs ci-dessus. En réalisant ce pairplot avec la séparation de couleurs entre vrais et faux billets, on remarque bien 2 groupes distincts pour chacune des variables, qui permettent certainement de différencier les billets falcifiés.
# 
# La longueur et la marge basse permettent le mieux de discriminer les vrais billets des faux.
# 

# Il ressort une corrélation linéaire entre les deux variables 'height_left' et 'height_right', le coefficient de pearson sera calculé pour vérifier l'observation.
# 
# Même si nous avons déjà les coefficients de Pearson dans le précédent HeatMap, une vérification est pertinente.
# 
# 

# In[32]:


#convert bool to int
df['is_genuine'] = df['is_genuine'].astype(int)


# In[33]:


plt.figure(figsize=(15,5))
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
plt.xticks(rotation=25, ha='right')
plt.title('Triangle de Corrélation',  fontsize=18, pad=20)
plt.savefig("graphiques/05. Triangle de corrélation.jpg", dpi=500, bbox_inches='tight', pad_inches=0.5)
    #plt.show()


# In[34]:


df.corr(method='pearson')


# Il existe donc une réelle corrélation linéaire entre height_right et height_left .car La valeur obtenue est proche de 1, Concrètement, au plus 'height_left' aura une valeur élevée, au plus 'height_right' le sera aussi.
# 
# La valeur obtenue est proche de -1, il existe donc une corrélation linéaire négative entre length et margin_low.

# In[ ]:





# In[ ]:




