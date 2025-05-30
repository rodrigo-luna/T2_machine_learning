import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parâmetros principais
ALEATORIO = True
SEMENTE = random.randint(0, 10000) if ALEATORIO else 69
TAMANHO_TESTE = 0.3
ALTURA_MAXIMA = 4    # None para tirar o limite

# Leitura dos dados de treino
colunas = ['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
df = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None, names=colunas, usecols=[3,4,5,6,7])

X = df[['qPA', 'pulso', 'resp']]
y = df['classe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TAMANHO_TESTE, random_state=SEMENTE)

# Inicia e treina o modelo
clf = DecisionTreeClassifier(criterion='entropy', max_depth=ALTURA_MAXIMA)
clf.fit(X_train, y_train)
# Faz previsões
y_prev = clf.predict(X_test)

# Avalia o modelo
precisao = accuracy_score(y_test, y_prev)
print(f"Precisão: {precisao * 100:.2f}%")

# Visualiza a árvore
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=['qPA', 'pulso', 'resp', 'classe'], filled=True)
plt.show()


# tree_rules = export_text(clf)
# print(tree_rules)