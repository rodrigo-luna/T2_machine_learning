import pandas as pd
import numpy as np

NOME_TXT = 'treino_sinais_vitais_com_label.txt'
COLUNAS = ['i', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
TOTAL_LINHAS = 1500
PORCENTAGEM_TREINO = 0.3
LINHAS_TREINO = int(TOTAL_LINHAS*PORCENTAGEM_TREINO)

train_data_m = pd.read_csv(NOME_TXT, header=None, names=COLUNAS, usecols=[3,4,5,6,7], nrows=LINHAS_TREINO)
test_data_m = pd.read_csv(NOME_TXT, header=None, names=COLUNAS, usecols=[3,4,5,6,7], skiprows=LINHAS_TREINO)

print(train_data_m.head())

# Calcula a entropia total do conjunto
def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0
    
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) 
        total_entr += total_class_entr
    
    print("Entropia total: ", total_entr)
    return total_entr

# Calcula a entropia de uma coluna
def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
    
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count
            entropy_class = - probability_class * np.log2(probability_class) 
        
        entropy += entropy_class
    return entropy

# Calcula o ganho de informação de um atributo
def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy
    return calc_total_entropy(train_data, label, class_list) - feature_info

# Calcula qual atributo gera mais ganho de informação
def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    print("Gerando sub-árvore ", feature_name)
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}
    
    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        
        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]

            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:
            tree[feature_value] = "?"
    return tree, train_data


def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(train_data, label, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        next_root = None
        
        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label, class_list)

def id3(train_data_m, label):
    print("Iniciando árvore...")
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data, label, class_list)
    
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None
        
def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]:
            correct_preditct += 1
        else:
            wrong_preditct += 1
    print ("Predições corretas: ", correct_preditct)
    print ("Predições erradas: ", wrong_preditct)
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    return accuracy

tree = id3(train_data_m, 'classe')

accuracy = evaluate(tree, test_data_m, 'classe')
print("Precisão:", accuracy)