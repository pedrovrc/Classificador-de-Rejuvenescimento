# Script para ordenar e salvar a lista de emails usados em treino de forma ordenada

# le numeros de arquivo e salva em lista
file = open("results/lista treino.txt")
numbers = []

for line in file:
    num = line.split()
    numbers.append(int(num[0]))
    
file.close()
# print(numbers)

# ordena numeros e salva em outro arquivo
numbers.sort()
# print(numbers)
file = open("results/lista treino ordenada.txt", "w")
for element in numbers:
    file.write(str(element) + '\n')