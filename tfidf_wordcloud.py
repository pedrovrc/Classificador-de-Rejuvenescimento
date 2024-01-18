# código feito com referência do artigo no link a seguir
# https://pub.towardsai.net/natural-language-processing-c12b0d525f99

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# leitura de arquivo de entrada
with open('text bodies/selected boost bodies.csv', encoding='utf8') as myfile:
    mydata = (line for line in myfile)
    database = pd.DataFrame(mydata, columns=['body'])
    
stopwords = set(STOPWORDS)
vectorizer = TfidfVectorizer()

# tratamento de texto de entrada
database["clean_text"] = database["body"].apply(lambda s: ' '.join(re.sub("[.,!?:;-='...'@#^></\|}{]", " ", s).split()))
database["clean_text"] = database["clean_text"].apply(lambda s: ' '.join(re.sub(r'\d+', '', s).split()))

def rem_en(input_txt):
    words = input_txt.lower().split()
    noise_free_words = [word for word in words if word not in stopwords] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

database["clean_text"] = database["clean_text"].apply(lambda s: rem_en(s))
lemmatiser = WordNetLemmatizer()
database["clean_text"] = database["clean_text"].apply(lambda row: [lemmatiser.lemmatize(r, pos='v') for r in row.split()])
database["clean_text"] = database["clean_text"].apply(lambda row: ' '.join(row))

# aplicação da TFIDF
response = vectorizer.fit_transform(database["clean_text"])
df_tfidf_sklearn = pd.DataFrame(response.toarray(), columns=vectorizer.get_feature_names_out())

# gera wordcloud com uso da TFIDF
tf_idf_counter = df_tfidf_sklearn.T.sum(axis=1)
wordcloud = WordCloud(width=3000, height=2000, background_color='white', stopwords=stopwords)
#wordcloud.generate(text)
wordcloud.generate_from_frequencies(tf_idf_counter)
wordcloud.to_file('wordcloud tfidf.png')

# mostra wordcloud
plt.figure(figsize=(40,30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()