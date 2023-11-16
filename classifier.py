# Classificador de posts sobre Rejuvenescimento de Código Fonte
# Este código foi feito utilizando as seguintes referências:
# Documentação do SGDClassifier do pacote Sci-Kit Learn: https://scikit-learn.org/stable/modules/sgd.html
# Artigo "Classificando textos com Machine Learning" no Medium: https://medium.com/luisfredgs/classificando-textos-com-machine-learning-e054ca7bf4e0



# IMPORTAÇÃO DE BIBLIOTECAS -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# vetorizador TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

# classificador linear utilizando gradiente descendente estocástico
from sklearn.linear_model import SGDClassifier
# rede neural perceptron multicamadas
from sklearn.neural_network import MLPClassifier

# métricas de avaliação
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



# TREINAMENTO DO MODELO ---------------------------------------------------------------------------
print("LENDO ARQUIVOS DE TREINO...")
# lê arquivo com emails selecionados
with open('base_treino_positivo.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    data_pos = pd.DataFrame(mydata, columns=['body'])
# marca como rejuvenescimento
data_pos['target'] = 1
data_pos['category'] = 'rejuvenation'

# mostra dataframe
# print(data_pos)
# print(type(data_pos))

# lê arquivo com emails que não são de rejuvenescimento
with open('base_treino_negativo.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    data_neg = pd.DataFrame(mydata, columns=['body'])
# marca como não rejuvenescimento
data_neg['target'] = 0
data_neg['category'] = 'not rejuvenation'

# mostra dataframe
# print(data_neg)
# print(type(data_neg))

# concatena ambos os dataframes e mostra
training_data = pd.concat([data_pos, data_neg])
# print(training_data)
# print(type(training_data))
print("LEITURA CONCLUÍDA.\n")

print("TREINANDO CLASSIFICADOR...")
# vetorização dos dados com a TFIDF
vectorizer = TfidfVectorizer()
X_train_tfidf_vectorize = vectorizer.fit_transform(training_data.loc[:, 'body'])

# Classificador
# Algoritmo: Stochastic Gradient Descent (SGD) - Gradiente Descendente Estocástico
# loss='hinge' -> equivalente a SVM linear (bom para poucos dados de treino)
clf = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-3, random_state = 42, max_iter = 5, tol = None)

# treino do classificador
clf.fit(X_train_tfidf_vectorize, training_data.loc[:, 'target'])
print("TREINAMENTO CONCLUÍDO.\n")



# TESTE E AVALIAÇÃO DO MODELO TREINADO ------------------------------------------------------------
print("LENDO ARQUIVOS DE TESTE...")
# lê arquivo de dados
with open('base_teste_data.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    test_data = pd.DataFrame(mydata, columns=['body'])
test_data['target'] = 0

# lê arquivo de classificações
with open('base_teste_class.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    test_class = pd.DataFrame(mydata, columns=['target'])

# atribui valor 'target' de test_class para test_data
for ind in test_class.index:
    if test_class.iloc[ind]['target'][0] == "1":
        test_data.at[ind, 'target'] = 1

# mostra dataframe
# print(test_data)
# print(type(test_data))
print("LEITURA CONCLUÍDA.\n")

print("AVALIANDO CLASSIFICADOR...")
# vetorização dos dados de teste
vect_transform = vectorizer.transform(test_data.loc[:, 'body'])

# predição com base no modelo
predicted = clf.predict(vect_transform)

# print(predicted)
# print(type(predicted))

print("MÉTRICAS OBTIDAS:")
# avalia e mostra métricas
print(metrics.classification_report(test_data.loc[:, 'target'], predicted, target_names = ['not rejuvenation', 'rejuvenation']))
print(clf.classes_)
confusion_matrix = confusion_matrix(test_data.loc[:, 'target'], predicted)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title("Matriz de confusão")
plt.colorbar()
plt.ylabel("Classificações corretas")
plt.xlabel("Classificações obtidas")
#plt.show()
print("AVALIAÇÃO CONCLUÍDA.\n")



# CLASSIFICAÇÃO DE NOVOS DADOS --------------------------------------------------------------------
print("LENDO DADOS PARA CLASSIFICAÇÃO...")
# obtém novos dados
with open('text bodies/boost bodies.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    new_data = pd.DataFrame(mydata, columns=['body'])

# mostra dataframe
# print(new_data)
# print(type(new_data))
print("LEITURA CONCLUÍDA.\n")

print("CLASSIFICANDO DADOS...")
# vetoriza novos dados e faz predição
X_new_tfidf_vectorize = vectorizer.transform(new_data.loc[:, 'body'])
predicted = clf.predict(X_new_tfidf_vectorize)

# print(predicted)
# print(type(predicted))
print("CLASSIFICAÇÃO CONCLUÍDA.")

# mostra indices de emails classificados como rejuvenescimento
# salva indices (id no banco) em txt
counter = 0
hits = 0
with open('results/boost rejuv.txt', 'w') as file:
    for classification in predicted:
        counter += 1
        if classification == 1:
            file.write(str(counter) + '\n')
            # print("Encontrado: índice ", counter)
            hits += 1

percentage = (hits/counter)*100
print("Entradas de rejuvenescimento encontradas:", hits, "de", counter, " (", round(percentage, 3), "% )")