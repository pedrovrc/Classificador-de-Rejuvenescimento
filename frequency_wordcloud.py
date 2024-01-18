# código feito com referência do artigo no link a seguir
# https://pub.towardsai.net/natural-language-processing-c12b0d525f99

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from collections import Counter
from nltk.stem import WordNetLemmatizer
import re

# leitura de arquivo de entrada
with open('text bodies/selected boost bodies.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    database = pd.DataFrame(mydata, columns=['body'])

# junta todos os dados em uma única string
text = " ".join(entry for entry in database['body'])
#print(text)

stopwords = set(STOPWORDS)

# obtém frequências de palavras
words = text.lower().split()
words = [re.sub("[.,!?:;-='...'@#^></\|}{]", " ", s) for s in words]
words = [re.sub(r'\d+', '', w) for w in words]
words = [word.strip() for word in words if word not in stopwords]
words.remove('')
lemmatiser = WordNetLemmatizer()
lem_words = [lemmatiser.lemmatize(w, pos='v') for w in words]
words_counter = Counter(lem_words)

# gera wordcloud
wordcloud = WordCloud(width=3000, height=2000, background_color='white', stopwords=stopwords)
#wordcloud.generate(text)
wordcloud.generate_from_frequencies(words_counter)
wordcloud.to_file('wordcloud freq.png')

# mostra wordcloud
plt.figure(figsize=(40,30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()