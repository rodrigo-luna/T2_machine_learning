import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Parâmetros principais
ALEATORIO = True
SEMENTE = random.randint(0, 10000) if ALEATORIO else 381
TAMANHO_TESTE = 0.3
N_ESTIMATORS = 100
N_SPLITS = 5

# Leitura dos dados de treino
colunas = ['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
df = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None, names=colunas)

X = df[['qPA', 'pulso', 'resp']]
y_reg = df['gravidade']
y_clf = df['classe']

# Validação cruzada
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEMENTE)

# Regressão
regressor_cv = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=SEMENTE)
mse_scores = cross_val_score(regressor_cv, X, y_reg, cv=cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)

print(f"\nValidação cruzada (Regressão - {N_SPLITS} folds):")
print("RMSE por fold:", np.round(rmse_scores, 4))
print("RMSE médio:", round(rmse_scores.mean(), 4))

# Classificação
classifier_cv = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEMENTE)
acc_scores = cross_val_score(classifier_cv, X, y_clf, cv=cv, scoring='accuracy')

print(f"\nValidação cruzada (Classificação - {N_SPLITS} folds):")
print("Acurácias por fold:", np.round(acc_scores * 100, 2), "%")
print("Acurácia média:", round(acc_scores.mean() * 100, 2), "%")

# Separação treino/teste
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=TAMANHO_TESTE, random_state=SEMENTE)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=TAMANHO_TESTE, random_state=SEMENTE)

# Regressor final
regressor = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=SEMENTE)
regressor.fit(X_train_r, y_train_r)
y_pred_r = regressor.predict(X_test_r)
rmse = sqrt(mean_squared_error(y_test_r, y_pred_r))
print("\nRMSE no conjunto de teste (Regressão):", round(rmse, 4))

# Classificador final
classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEMENTE)
classifier.fit(X_train_c, y_train_c)
y_pred_c = classifier.predict(X_test_c)

print("\nRelatório de Classificação:")
report = classification_report(y_test_c, y_pred_c, output_dict=True)
print(f"{'Classe':<8} {'Precisão':>10} {'Revocação':>12} {'F1-score':>12} {'Suporte':>10}")
for classe, m in report.items():
    if classe in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    print(f"{classe:<8} {m['precision']*100:10.2f} {m['recall']*100:12.2f} {m['f1-score']*100:12.2f} {int(m['support']):10}")

print(f"\nAcurácia geral no teste: {report['accuracy']*100:.2f}%")

print("\nMatriz de confusão:")
print(confusion_matrix(y_test_c, y_pred_c))

# Importância das variáveis
feat_names = X.columns

# Regressão
importancias_r = regressor.feature_importances_
df_r = pd.DataFrame({'Variável': feat_names, 'Importância': importancias_r})
df_r = df_r.sort_values(by='Importância', ascending=False)

print("\nImportância das variáveis (Regressão):")
print(df_r)

plt.figure(figsize=(6, 4))
sns.barplot(x='Importância', y='Variável', data=df_r)
plt.title('Importância - Regressão')
plt.tight_layout()
plt.show()

# Classificação
importancias_c = classifier.feature_importances_
df_c = pd.DataFrame({'Variável': feat_names, 'Importância': importancias_c})
df_c = df_c.sort_values(by='Importância', ascending=False)

print("\nImportância das variáveis (Classificação):")
print(df_c)

plt.figure(figsize=(6, 4))
sns.barplot(x='Importância', y='Variável', data=df_c)
plt.title('Importância - Classificação')
plt.tight_layout()
plt.show()

# Teste cego
colunas_teste = ['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade_dummy']
df_teste = pd.read_csv('treino_sinais_vitais_sem_label.txt', header=None, names=colunas_teste)
X_teste = df_teste[['qPA', 'pulso', 'resp']]

pred_gravidade = regressor.predict(X_teste)
pred_classe = classifier.predict(X_teste)

# Saída final
saida = pd.DataFrame({
    'i': df_teste['i'].astype(int),
    'gravidade': pred_gravidade,
    'classe': pred_classe.astype(int)
})
saida.to_csv('saida_teste_cego.csv', index=False, float_format='%.4f')
print("\nArquivo 'saida_teste_cego.csv' salvo.")

# Gráficos
# 1. Regressão: Real vs. Previsto
plt.figure(figsize=(6, 6))
plt.scatter(y_test_r, y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.title('Real vs. Previsto - Regressão')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Resíduos
residuos = y_test_r - y_pred_r
plt.figure(figsize=(8, 4))
plt.hist(residuos, bins=30, color='orange', edgecolor='black')
plt.title('Distribuição dos resíduos')
plt.xlabel('Erro')
plt.ylabel('Frequência')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_c, y_pred_c), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.show()
