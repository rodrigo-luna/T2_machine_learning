import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import seaborn as sns

# Parâmetros de treinamento
NUM_EPOCAS = 100000
TAXA_APRENDIZADO = 0.001
CAMADAS_ESCONDIDAS = [256, 128, 64]
TAMANHO_TESTE = 0.3

# Arquivo de entrada
NOME_TXT = 'treino_sinais_vitais_com_label.txt'
SEMENTE = random.randint(0, 1000)

# ========================
# MLP para Classificação
# ========================
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i - 1]) * np.sqrt(2. / sizes[i - 1]))
            self.biases.append(np.zeros((sizes[i], 1)))

    def feed_forward(self, X):
        self.activations = [X]
        self.z = []
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            a = self.relu(z) if i < self.num_layers - 1 else self.softmax(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[1]
        dZ = self.activations[-1] - y
        gradients = []
        for i in range(self.num_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))
            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
                dZ = dA * self.gradient_relu(self.z[i - 1])
        return gradients[::-1]

    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def relu(self, Z):
        return np.maximum(0, Z)

    def gradient_relu(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# ========================
# MLP para Regressão
# ========================
class MLPRegressor:
    def __init__(self, input_size, hidden_sizes, output_size=1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i - 1]) * np.sqrt(2. / sizes[i - 1]))
            self.biases.append(np.zeros((sizes[i], 1)))

    def feed_forward(self, X):
        self.activations = [X]
        self.z = []
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            a = self.relu(z) if i < self.num_layers - 1 else z  # Linear na última camada
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[1]
        dZ = self.activations[-1] - y
        gradients = []
        for i in range(self.num_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))
            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
                dZ = dA * self.gradient_relu(self.z[i - 1])
        return gradients[::-1]

    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def relu(self, Z):
        return np.maximum(0, Z)

    def gradient_relu(self, Z):
        return (Z > 0).astype(float)

# Leitura dos dados
colunas = ['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
df = pd.read_csv(NOME_TXT, header=None, names=colunas, usecols=[3, 4, 5, 6, 7])
X = df[['qPA', 'pulso', 'resp']]
y_class = df['classe']
y_reg = df['gravidade']

# Divisão
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=TAMANHO_TESTE, random_state=SEMENTE
)

# Normalização
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train_norm = (X_train - X_train_mean) / X_train_std
X_test_norm = (X_test - X_train_mean) / X_train_std

# Dados para MLP
X_train_np = X_train_norm.to_numpy().T
X_test_np = X_test_norm.to_numpy().T
y_class_train_oh = np.eye(4)[y_class_train.to_numpy() - 1].T
y_class_test_oh = np.eye(4)[y_class_test.to_numpy() - 1].T
y_reg_train_np = y_reg_train.to_numpy().reshape(1, -1)
y_reg_test_np = y_reg_test.to_numpy().reshape(1, -1)

# Instanciar modelos
mlp_class = MLP(3, CAMADAS_ESCONDIDAS, 4)
mlp_reg = MLPRegressor(3, CAMADAS_ESCONDIDAS)

historico_class = []
for epoch in range(NUM_EPOCAS):
    # Classificação
    out_class = mlp_class.feed_forward(X_train_np)
    grad_class = mlp_class.backward(X_train_np, y_class_train_oh)
    mlp_class.update_parameters(grad_class, TAXA_APRENDIZADO)
    loss_class = -np.mean(np.sum(y_class_train_oh * np.log(out_class + 1e-9), axis=0))

    if (epoch + 1) % 1000 == 0:
        historico_class.append(loss_class)
        print(f"Época {epoch + 1} - Erro de Classificação (Cross-Entropy): {loss_class:.4f}")

# === Teste da Classificação ===
print("\nVerificando grupo de teste...")
test_outputs = mlp_class.feed_forward(X_test_np)
test_loss = -np.mean(np.sum(y_class_test_oh * np.log(test_outputs + 1e-9), axis=0))
print(f"Erro de teste (cross-entropy): {test_loss:.4f}")

predictions = np.argmax(test_outputs, axis=0) + 1
true_labels = y_class_test.to_numpy()
accuracy = np.mean(predictions == true_labels)
print(f"Acurácia no teste: {accuracy:.2%}")

plt.figure(figsize=(15, 8))
plt.plot(historico_class, color='black')
plt.title('Treinamento - Erro (Cross-Entropy)')
plt.xlabel('Épocas x100')
plt.ylabel('Erro')
plt.grid()
plt.show()

print("\nRelatório de Classificação:")
report = classification_report(true_labels, predictions, output_dict=True)
print(f"{'Classe':<8} {'Precisão':>10} {'Revocação':>12} {'F1-score':>12} {'Suporte':>10}")
for classe, m in report.items():
    if classe in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    print(f"{classe:<8} {m['precision']*100:10.2f} {m['recall']*100:12.2f} {m['f1-score']*100:12.2f} {int(m['support']):10}")
print(f"\nAcurácia geral no teste: {report['accuracy']*100:.2f}%")

print("\nMatriz de confusão:")
cm = confusion_matrix(true_labels, predictions)
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - MLP com ReLU')
plt.tight_layout()
plt.show()

# === Regressão ===
print("\nTreinando regressão de gravidade...")
for epoch in range(NUM_EPOCAS):
    pred = mlp_reg.feed_forward(X_train_np)
    grad = mlp_reg.backward(X_train_np, y_reg_train_np)
    mlp_reg.update_parameters(grad, TAXA_APRENDIZADO)

# Teste Regressão
y_pred_r = mlp_reg.feed_forward(X_test_np).flatten()
y_test_r = y_reg_test.to_numpy()
mse = mean_squared_error(y_test_r, y_pred_r)
print(f"\nErro quadrático médio (MSE) da regressão: {mse:.4f}")

# 1. Regressão: Real vs. Previsto (gráfico solicitado)
plt.figure(figsize=(6, 6))
plt.scatter(y_test_r, y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.title('Real vs. Previsto - Regressão')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Teste Cego ===
df_teste = pd.read_csv('treino_sinais_vitais_sem_label.txt', header=None,
                       names=['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade_dummy'])

X_teste = df_teste[['qPA', 'pulso', 'resp']]
X_teste_norm = (X_teste - X_train_mean) / X_train_std
X_teste_np = X_teste_norm.to_numpy().T

# Previsões
pred_class_cego = np.argmax(mlp_class.feed_forward(X_teste_np), axis=0) + 1
pred_gravidade_cego = mlp_reg.feed_forward(X_teste_np).flatten()

saida = pd.DataFrame({
    'i': df_teste['i'].astype(int),
    'classe': pred_class_cego.astype(int),
    'gravidade_prevista': pred_gravidade_cego
})
saida.to_csv('saida_teste_cego_MLP.csv', index=False)
print("\nArquivo 'saida_teste_cego_MLP.csv' salvo com as previsões da rede neural.")
