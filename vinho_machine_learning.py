import pandas as pd
arquivo = pd.read_csv('wine_dataset.csv')

# Troque a palavra red por 0, e white por 1:
arquivo["style"] = arquivo['style'].replace('red', 0)
arquivo["style"] = arquivo['style'].replace('white', 1)

#print(arquivo.head)

# Separando as variáveis em preditoras e variável alvo:
y = arquivo['style'] # y é a coluna style
x = arquivo.drop('style', axis=1) # x fica sendo todas as colunas menos a style.

# Importe a função train_test_split para dividir os dados em dados de treino e 
#   dados de teste:
from sklearn.model_selection import train_test_split

#criando conjuntos de dados de treino e teste:
# 0.3 = 30% dos dados são para teste, que foram dividos aleatoriamente!
# 70% são dados para treino. 
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3) 

# Importe o modelo de classificação que vamos utilizar: 
from sklearn.ensemble import ExtraTreesClassifier

#criação do modelo
modelo = ExtraTreesClassifier()

# treine o modelo:
modelo.fit(x_treino, y_treino)

# Acurácia do modelo: 
resultado = modelo.score(x_teste, y_teste)
print('Acurácia', resultado) 

# Vamos verificar se o nosso modelo consegue prever um resultado:
previsao = modelo.predict(x_teste[400:403])
print("previsão =", previsao)

# Resultado esperado: 
print("resultado esperado:\n", y_teste[400:403])
