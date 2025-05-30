import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parâmetros principais
ALEATORIO = True
SEMENTE = random.randint(0, 10000) if ALEATORIO else 69
TAMANHO_TESTE = 0.5
ALTURA_MAXIMA = None    # None para tirar o limite

# Leitura dos dados de treino
colunas = ['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
df = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None, names=colunas, usecols=[3,4,5,6,7])

X = df[['qPA', 'pulso', 'resp']]
y = df['classe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TAMANHO_TESTE, random_state=SEMENTE)

# Inicia e constrói o modelo
clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0, max_depth=ALTURA_MAXIMA)
clf.fit(X_train, y_train)

# Testa o modelo
y_prev = clf.predict(X_test)

# Avalia o modelo
precisao = accuracy_score(y_test, y_prev)
print(f"Precisão: {precisao * 100:.2f}%")

# Visualiza a árvore
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=['qPA', 'pulso', 'resp'], filled=True)
plt.show()