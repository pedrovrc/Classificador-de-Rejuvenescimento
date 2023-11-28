# PROGRAMA MODIFICADO, ORIGINAL NO COLAB: https://colab.research.google.com/drive/1pcNl7OjO-OZ65eJYboQL0UOVrpn0ImRE

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from wordcloud import WordCloud
from tabulate import tabulate
import matplotlib.pyplot as plt

# download de recursos do NLTK
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('omw')
nltk.download('wordnet')

# instancia classes do NLTK
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()



# FUNÇÕES AUXILIARES ------------------------------------------------------------------------------
# Função de normalização de texto
def normalize_text(text):
    # Tokenização
    tokens = word_tokenize(text)

    # Remoção de stopwords e caracteres não alfanuméricos
    tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]

    # POS tagging
    pos_tags = pos_tag(tokens)

    # NER tagging
    ner_tags = ne_chunk(pos_tags)

    # Lematização
    lemmas = [lemmatizer.lemmatize(token, pos=tag[0].lower())
    if tag[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(token) for token, tag in zip(tokens, pos_tags)]

    return ' '.join(lemmas)

# Função para determinar o tipo de sentimento
def sentiment_type(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'
    
# Função para exibir a tabela de dados
def show_table(data, line_count):
    headers = ['Text', 'Sentiment', 'Type']
    table = data.head(line_count)
    formatted_table = tabulate(table, headers=headers, tablefmt='psql')
    print(formatted_table)
    
# Função para exibir a quantidade de posts por tipo de sentimento
def show_quantities(data):
    positives = data[data['sentiment_type'] == 'positive'].shape[0]
    neutrals = data[data['sentiment_type'] == 'neutral'].shape[0]
    negatives = data[data['sentiment_type'] == 'negative'].shape[0]

    print("---------------------------------------------------------------------\n\n\n")
    print("Quantidade de posts positivos:", positives)
    print("Quantidade de posts neutros:", neutrals)
    print("Quantidade de posts negativos:", negatives)
    print("---------------------------------------------------------------------\n\n\n")

    # Configuração do gráfico
    tipos = ['Positivos', 'Neutros', 'Negativos']
    quantidades = [positives, neutrals, negatives]

    plt.bar(tipos, quantidades)
    plt.title('Quantidade de Posts por Sentimentos')
    plt.xlabel('Sentimentos')
    plt.ylabel('Quantidades')
    plt.show()



# INICIO DO PROGRAMA ------------------------------------------------------------------------------
# Carrega stopwords
stop_words = set(stopwords.words('english'))

# Carrega dados do arquivo
with open('text bodies/selected boost bodies.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    database = pd.DataFrame(mydata, columns=['body'])

# Trata valores ausentes
database = database.dropna()

# Pré-processamento de texto: Normalização
database['processed_text'] = database['body'].apply(normalize_text)

# Cria nova coluna para armazenar as pontuações de sentimento
database['sentiment'] = database['body'].apply(lambda text: sia.polarity_scores(text)['compound'])

# Seleciona apenas as colunas 'body' e 'sentiment', adiciona coluna 'sentiment_type'
selected_data = database[['body', 'sentiment']]
selected_data['sentiment_type'] = selected_data['sentiment'].apply(sentiment_type)



# EXIBIÇÃO DE RESULTADOS --------------------------------------------------------------------------
# Exibe tabela de dados e quantidades de posts por tipo de sentimento
show_table(selected_data, line_count=10)
show_quantities(selected_data)
print("---------------------------------------------------------------------\n\n\n")

# Combina todos os textos em um único string e tokeniza
combined_text = ' '.join(selected_data['body'])
tokens = word_tokenize(combined_text)

# Cria a nuvem de palavras apenas com as palavras filtradas
text = ' '.join(tokens)
STOPWORDS = set([])
wordcloud = WordCloud(width=1600, height=800, random_state=42, collocations=True, scale=2, stopwords=STOPWORDS, prefer_horizontal=0.9).generate(text)

# Plota a nuvem de palavras
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()