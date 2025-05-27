# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# import matplotlib.pyplot as plt
# import math

# CSV_NAME = 'treino.csv'

# def calc_entropia(data, target_column):
#     total_rows = len(data)
#     target_values = data[target_column].unique()

#     entropy = 0
#     for value in target_values:
#         # Calcula a proporção de casos com o valor atual
#         value_count = len(data[data[target_column] == value])
#         proportion = value_count / total_rows
#         entropy -= proportion * math.log2(proportion)

#     return entropy

# def calc_ganho_de_informacao(data, feature, target_column):

#     # Calcula a entropia média ponderada para o atributo
#     unique_values = data[feature].unique()
#     weighted_entropy = 0

#     for value in unique_values:
#         subset = data[data[feature] == value]
#         proportion = len(subset) / len(data)
#         weighted_entropy += proportion * calc_entropia(subset, target_column)

#     # Calcula o ganho de informação
#     information_gain = entropy_g - weighted_entropy

#     return information_gain

# def id3(data, target_column, features):
#     if len(data[target_column].unique()) == 1:
#         return data[target_column].iloc[0]

 
#     if len(features) == 0:
#         return data[target_column].mode().iloc[0]

#     best_feature = max(features, key=lambda x: calc_ganho_de_informacao(data, x, target_column))

#     tree = {best_feature: {}}

#     features = [f for f in features if f != best_feature]

#     for value in data[best_feature].unique():
#         subset = data[data[best_feature] == value]
#         tree[best_feature][value] = id3(subset, target_column, features)

#     return tree


# df = pd.read_csv(CSV_NAME)
# df.head()

# entropy_g = calc_entropia(df, 'g')
# print(f'Entropy of the dataset: {entropy_g}')

# for column in df.columns[:-1]:
#     entropy = calc_entropia(df, column)
#     information_gain = calc_ganho_de_informacao(df, column, 'g')
#     print(f'{column} - Entropy: {entropy:.3f}, Information Gain: {information_gain:.3f}')

# # Feature selection for the first step in making decision tree
# selected_feature = 's1'

# # Create a decision tree
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
# X = df[[selected_feature]]
# y = df['g']
# clf.fit(X, y)

# plt.figure(figsize=(8, 6))
# plot_tree(clf, feature_names=[selected_feature], class_names=['0', '1'], filled=True, rounded=True)
# plt.show()