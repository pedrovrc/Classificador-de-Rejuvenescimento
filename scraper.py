# Script para verificar a quantidade de mensagens numa lista de mensagens do OpenJDK.
# Em cada página de mensagens separadas por mês, existe um número de mensagens total daquele mês.
# Esse script soma todos esses números e apresenta o resultado no final.

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

# Abre conexão para o site
# Alterar a URL base para calcular para outras listas.
base_url = "https://mail.openjdk.org/pipermail/hotspot-dev/"
page = urlopen(base_url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")

# Encontrar lista com todos os links para as mensagens de cada mês, organizadas por data:
table = soup.find("table")
skip_first = True
url_list = []
for tr in table.find_all("tr"):
    if (skip_first) :   # Pula primeiro elemento (cabeçalho)
        skip_first = False
        continue
    td_tag = tr.find(href=re.compile("date.html"))
    date_url = td_tag.get('href')
    url_list.append(base_url + date_url)
#print(url_list) # debug

# Contar quantas mensagens estão presentes:
quantity = 0
for url in url_list:
    print("URL atual:", url)
    
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    
    p_tags = soup.find("p")
    b_tag = p_tags.find("b", string=re.compile("Messages:"))
    quantity += int(b_tag.next_sibling.strip())
    
    print("Quantidade nessa URL:", int(b_tag.next_sibling.strip()))
    print("Soma atual:", quantity, "\n")

print("Quantidade total:", quantity)