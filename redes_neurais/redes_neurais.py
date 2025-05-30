import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Parâmetros de treinamento
NUM_EPOCAS = 50000
TAXA_APRENDIZADO = 0.05
CAMADAS_ESCONDIDAS = [5, 5]         # Número de camadas / número de neurônios por camada
TAMANHO_TESTE = 0.5                 # Porcentagem dos dados que serão usados para teste 
# BATELADA = 5                      # Para o backpropagation

# Parâmetros de leitura do arquivo
NOME_TXT = 'treino_sinais_vitais_com_label.txt'
SEMENTE = random.randint(0, 1000)

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        # Inicia os pesos e bias de cada camada com valores aleatórios
        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]))
            self.biases.append(np.random.randn(sizes[i], 1))

    # Propagação para frente
    def feed_forward(self, X):
        self.activations = [X]
        self.z = []
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            if i < self.num_layers - 1:
                a = self.tanh(z)  # Ativação com Tangente Hiperbólica nas camadas escondidas
            else:
                a = z  # Ativação linear na camada final
            self.activations.append(a)
        return self.activations[-1]  # shape: (output_size, m)

    # Backpropagation
    def backward(self, X, y):
        m = X.shape[1]  # Número de linhas de treinamento

        gradients = []
        dZ = self.activations[-1] - y  # shape: (output_size, m)
        for i in range(self.num_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)  # shape: (sizes[i], sizes[i-1])
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # shape: (sizes[i], 1)
            gradients.append((dW, db))
            
            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)  # shape: (sizes[i-1], m)
                dZ = dA * self.gradient_tanh(self.z[i-1])  # shape: (sizes[i-1], m)

        return gradients[::-1]  # Reverse the gradients

    def update_parameters(self, gradients, learning_rate):
        # Atualiza os pesos e biases de acordo com o cálculo anterior
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def tanh(self, Z):
        # Função de ativação por Tangente Hiperbólica (tanh)
        return np.tanh(Z)

    def gradient_tanh(self, Z):
        # Função de gradiente da tanh
        return 1 - np.tanh(Z)**2


# Leitura dos dados de treino
colunas = ['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
df = pd.read_csv(NOME_TXT, header=None, names=colunas, usecols=[3,4,5,6,7])
print(df.head())

X = df[['qPA', 'pulso', 'resp', 'gravidade']]
y = df['classe']
# Separação dos dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TAMANHO_TESTE, random_state=SEMENTE)

# Normalizando os dados de entrada
X_train_mean = np.mean(X_train)
X_train_std = np.std(X_train)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# Transformando em colunas numpy
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

# Definindo o MLP
input_size = X_train.shape[1]
output_size = y_train.shape[1]
mlp = MLP(input_size, CAMADAS_ESCONDIDAS, output_size)

# Treinamento
for epoch in range(NUM_EPOCAS):
    outputs = mlp.feed_forward(X_train.T)

    # Faz o backpropagation e atualiza os pesos e bias a cada batelada

    gradients = mlp.backward(X_train.T, y_train.T)
    mlp.update_parameters(gradients, TAXA_APRENDIZADO)

    # Calcula o erro
    loss = np.mean((outputs - y_train.T) ** 2)
    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch+1} - Taxa de erro: {loss}")
print("Fim do treinamento.")

# Testando
print("Verificando grupo de teste...")
test_outputs = mlp.feed_forward(X_test.T)
test_loss = np.mean((test_outputs - y_test.T) ** 2)
print(f"Taxa de erro do teste: {test_loss}")