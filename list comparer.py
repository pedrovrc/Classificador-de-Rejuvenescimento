# Script para marcar e analisar dados obtidos do classificador sobre o dataset Boost

# abre arquivos de entrada
training_file = open("results/lista treino ordenada.txt")   # lista de emails contidos nos datasets de treino e teste do classificador
classification_file = open("boost rejuv.txt")   # lista de emails classificados como rejuvenescimento de código fonte (resultado do classificador)

training_list = []
classification_list = []

# conta quantidade e armazena indices da lista de treino
train_count = 0
for line in training_file:
    num = line.split()
    training_list.append(int(num[0]))
    train_count += 1

# conta quantidade e armazena indices da lista de classificados
class_count = 0
for line in classification_file:
    num = line.split()
    classification_list.append(int(num[0]))
    class_count += 1

# cria arquivo novo e marca elementos presentes em ambas as listas
# com esse arquivo é possível criar um subset de emails classificados no banco no SQL.
# basta filtrar pelos índices, que correspondem à chave primária "id" no banco.
# caso não queira um arquivo contendo os índices marcados como treino, basta comentar a linha de marcação.
collision_count = 0
revised_file = open("results/boost rejuv revised.txt", "w")
for element in classification_list:
    if element in training_list: 
        revised_file.write(str(element) + " - treino\n")    # marcação
        collision_count += 1
    else:
        revised_file.write(str(element) + "\n")
        
# mostra métricas obtidas
print("number of training data entries: ", train_count)
print("number of classified data entries: ", class_count)
print("number of collisions: ", collision_count)
print("number of new cases found: ", class_count - collision_count)