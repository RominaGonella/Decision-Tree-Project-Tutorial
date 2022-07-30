### DECISION TREE PROJECT TUTORIAL

# Step 1: cargar los datos y guardarlos

# Como es habitual, primero ejecutar en consola `pip install -r requirements.txt`, además puede ser necesario:
#pip install pandas
#pip install matplotlib
#pip install seaborn
#pip install sklearn

# importamos librerías
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

# cargamos datos
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

# se cambia a categórico
df_raw.Outcome = pd.Categorical(df_raw.Outcome)
# se guardan datos iniciales
df_raw.to_csv('data/raw/datos_iniciales.csv', index = False)

# separo en X e y
X = df_raw.iloc[:, :8]
y = df_raw.iloc[:, 8]

# separo en muestras de entrenamiento y evaluación, dejo la proporción de test por defecto (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3007)

# defino datset a realizar análisis exploratorio
df_train = pd.concat([X_train, y_train], axis = 1)
df_train

# descarto observaciones anómalas
anom = (df_train.Glucose == 0) | (df_train.BloodPressure == 0) | (df_train.BMI == 0)
df_train = df_train[anom == False]

# vuelvo a definir X_train y y_train para eliminar esos anómalos
X_train = df_train.iloc[:, :8]
y_train = df_train.iloc[:, 8]

# guardo datos procesados
df_train.to_csv('data/processed/datos_entrenamiento_procesados.csv', index = False)

# creo modelo final (con hiperparámetros optimizados en notebook explore.ipynb)
clf_best = DecisionTreeClassifier(random_state = 3007, criterion = 'gini', max_depth = 5, min_samples_leaf = 1, min_samples_split =  2)
clf_best.fit(X_train, y_train)

# se guarda modelo final
filename = 'models/decTree_model.sav'
pickle.dump(clf_best, open(filename,'wb'))