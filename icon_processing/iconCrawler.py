from bs4 import BeautifulSoup
#import pandas as pd

result = []
print('icons crawling >>>>>>>>>>>>>>>>>>>>>>>>>>')

icon_txt = 'iconList.txt'
print(icon_txt)

file = open(icon_txt, encoding='utf-8').read()
soup = BeautifulSoup(file, 'html.parser')

for iconList in soup.find_all('div', class_='emoji_card_list'):
    i_tag = iconList.find('i')
    i_id = i_tag['id']
    iconCategory = i_id[-1]
    if iconCategory in ('A', 'D', 'E', 'F', 'G', 'H'):
        for iconCard in iconList.find_all('div', class_='emoji_card'):
            iconFont = iconCard.find('a', class_='emoji_font').string
            iconName = iconCard.find('a', class_='emoji_name').string
            result.append([iconFont, iconName, iconCategory])
          
sqlFile = open('icon_sql.txt', 'w', encoding='utf-8')
iconId = 1
for (font, name, category) in result:
    query = str(iconId) + ". INSERT INTO Icon(font, name, category) VALUES ('" + font + "', '" + name + "', '" + category + "'); \n"
    sqlFile.write(query)
    iconId+=1
sqlFile.close()