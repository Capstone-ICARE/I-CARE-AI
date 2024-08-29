import pandas as pd

icons = pd.read_csv('icon.csv')
newIcons = []
newId = 1

A_remove_iconId = [10, 12, 14, 19, 20, 22, 25, 26, 27, 29, 30, 31, 34, 35, 36, 37, 41, 42, 43, 44, 49, 50, 53, 58, 59, 63, 67, 69, 70, 71, 72, 77, 79, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 152, 153, 155, 156, 157, 158, 159, 160, 161]

A_fun_iconId = [1, 2, 3, 4, 5, 7, 9, 13, 24, 68]
A_love_iconId = [15, 16, 17, 18, 21, 129, 131]
A_surprise_iconId = [28, 32, 78, 80]
A_tense_iconId = [51, 62, 65, 75, 81, 86, 87, 91, 92]
A_disappoint_iconId = [38, 39, 40, 48, 74, 76, 84, 85, 94]
A_angry_iconId = [47, 66, 99, 100, 101, 151]
A_sad_iconId = [73, 89, 90, 93]
A_tired_iconId = [11, 45, 52, 54, 96, 97, 98, 154, 162]
A_embarrassed_iconId = [6, 8, 33, 46]
A_envy_iconId = [23, 82, 83, 88, 95]
A_hurt_iconId = [55, 56, 57, 60, 61, 64]

A_icons = icons[icons['category'] == 'A']

for index, icon in A_icons.iterrows():
    iconId = icon['iconId']
    if iconId not in A_remove_iconId and not (132 <= iconId <= 150):
        name = icon['name']
        if iconId in A_fun_iconId:
            name = '재밌다'
        elif iconId in A_love_iconId:
            name = '사랑하다'
        elif iconId in A_surprise_iconId:
            name = '놀라다'
        elif iconId in A_tense_iconId:
            name = '긴장하다'
        elif iconId in A_disappoint_iconId:
            name = '실망하다'
        elif iconId in A_angry_iconId:
            name = '화나다'
        elif iconId in A_sad_iconId:
            name = '슬프다'
        elif iconId in A_tired_iconId:
            name = '피곤하다'
        elif iconId in A_embarrassed_iconId:
            name = '민망하다'
        elif iconId in A_envy_iconId:
            name = '아쉽다'
        elif iconId in A_hurt_iconId:
            name = '아프다'
        newIcons.append([newId, icon['font'], name, icon['category']])
        newId += 1
        
        
D_remove_iconId = [163, 167, 169, 170, 171, 176, 177, 180, 183, 187, 189, 190, 191, 193, 195, 196, 200, 208, 209, 212, 223, 229, 230, 232, 251, 252, 254, 283, 285, 287, 290, 304, 305]
D_icons = icons[icons['category'] == 'D']
for index, icon in D_icons.iterrows():
    iconId = icon['iconId']
    if iconId not in D_remove_iconId:
        name = icon['name']
        if iconId == 253:
            name = '공룡'
        elif iconId == 286:
            name = '장미'
        elif iconId == 306:
            name = '둥지'
        else:
            name = name.replace(' 얼굴', '')
        newIcons.append([newId, icon['font'], name, icon['category']])
        newId += 1
        
        
E_remove_iconId = [317, 353, 363, 367, 370, 392, 398, 399, 403, 405, 415, 417, 418, 419, 420, 422, 423, 424, 425, 426, 427, 429, 430, 434, 435, 436, 437, 438]
E_icons = icons[icons['category'] == 'E']
for index, icon in E_icons.iterrows():
    iconId = icon['iconId']
    if iconId not in E_remove_iconId:
        name = icon['name']
        if iconId == 316:
            name = '사과'
        elif iconId == 335:
            name = '채소'
        elif iconId == 350:
            name = '치즈'
        elif iconId == 352:
            name = '치킨'
        elif iconId == 371:
            name = '샐러드'
        elif iconId == 380:
            name = '카레'
        elif iconId == 391:
            name = '포춘 쿠키'
        elif iconId == 404:
            name = '케이크'
        elif iconId == 410:
            name = '푸딩'
        elif iconId == 413:
            name = '우유'
        elif iconId == 414:
            name = '커피'
        elif iconId == 433:
            name = '외식'
        newIcons.append([newId, icon['font'], name, icon['category']])
        newId += 1
        
        
F_remove_iconId = [439, 440, 442, 446, 453, 456, 462, 463, 465, 467, 477, 490, 491, 492, 493, 495, 496, 497, 515, 516, 518, 519, 520, 524, 526, 528, 530, 531, 534, 538, 546, 547, 552, 553, 554, 565, 566, 567, 571, 572, 577, 579, 582, 584, 585, 619, 620, 621, 623, 624, 628, 642, 645, 646, 648, 649, 651, 653]
F_icons = icons[icons['category'] == 'F']
for index, icon in F_icons.iterrows():
    iconId = icon['iconId']
    if iconId not in F_remove_iconId and not (586 <= iconId <= 617) and not (632 <= iconId <= 636):
        name = icon['name']
        if iconId == 441:
            name = '지구'
        elif iconId == 443:
            name = '지도'
        elif iconId == 444:
            name = '일본'
        elif iconId == 451:
            name = '해변'
        elif iconId == 457:
            name = '공사'
        elif iconId == 478:
            name = '성'
        elif iconId == 523:
            name = '경찰'
        elif iconId == 527:
            name = '차'
        elif iconId == 532:
            name = '트럭'
        elif iconId == 537:
            name = '휠체어'
        elif iconId == 544:
            name = '정류장'
        elif iconId == 551:
            name = '신호등'
        elif iconId == 557:
            name = '배'
        elif iconId == 578:
            name = '여행 가방'
        elif iconId == 580:
            name = '모래시계'
        elif iconId == 618:
            name = '달'
        elif iconId == 625:
            name = '해'
        elif iconId == 626:
            name = '행성'
        elif iconId == 637:
            name = '비'
        elif iconId == 638:
            name = '눈'
        elif iconId == 639:
            name = '번개'
        elif iconId == 647:
            name = '우산'
        elif iconId == 652:
            name = '눈사람'
        elif iconId == 655:
            name = '물'
        newIcons.append([newId, icon['font'], name, icon['category']])
        newId += 1
        

G_remove_iconId = [662, 675, 676, 678, 696, 715, 718, 719, 723, 724, 725, 732, 738, 741]
G_icons = icons[icons['category'] == 'G']
for index, icon in G_icons.iterrows():
    iconId = icon['iconId']
    if iconId not in G_remove_iconId and not (665 <= iconId <= 671):
        name = icon['name']
        if iconId == 658:
            name = '트리'
        elif iconId == 680:
            name = '메달'
        elif iconId == 684:
            name = '축구'
        elif iconId == 685:
            name = '야구'
        elif iconId == 688:
            name = '배구'
        elif iconId == 689:
            name = '미식축구'
        elif iconId == 690:
            name = '럭비'
        elif iconId == 695:
            name = '하키'
        elif iconId == 700:
            name = '권투'
        elif iconId == 705:
            name = '낚시'
        elif iconId == 706:
            name = '다이빙'
        elif iconId == 710:
            name = '컬링'
        elif iconId == 711:
            name = '과녁'
        elif iconId == 716:
            name = '마술'
        elif iconId == 717:
            name = '게임'
        elif iconId == 722:
            name = '곰인형'
        elif iconId == 729:
            name = '클로버'
        elif iconId == 730:
            name = '체스'
        elif iconId == 736:
            name = '그림'
        elif iconId == 740:
            name = '실'
        newIcons.append([newId, icon['font'], name, icon['category']])
        newId += 1


H_remove_iconId = [757, 758, 760, 761, 768, 771, 772, 778, 780, 781, 782, 783, 795, 796, 797, 799, 800, 801, 835, 836, 837, 840, 844, 848, 859, 861, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 876, 889, 891, 897, 898, 899, 900, 913, 917, 918, 921, 922, 923, 925, 930, 931, 951, 952, 953, 954, 957, 970]        
H_icons = icons[icons['category'] == 'H']
for index, icon in H_icons.iterrows():
    iconId = icon['iconId']
    if iconId not in H_remove_iconId and not (787 <= iconId <= 793) and not (815 <= iconId <= 822) and not (827 <= iconId <= 833) and not (849 <= iconId <= 855) and not (879 <= iconId <= 887) and not (902 <= iconId <= 911) and not (940 <= iconId <= 948) and not (990 <= iconId <= 998):
        name = icon['name']
        if iconId == 745:
            name = '실험실'
        elif iconId == 766:
            name = '가방'
        elif iconId == 774:
            name = '발레'
        elif iconId == 777:
            name = '모자'
        elif iconId == 779:
            name = '졸업'
        elif iconId == 786:
            name = '보석'
        elif iconId == 798:
            name = '노래'
        elif iconId == 813:
            name = '북'
        elif iconId == 824:
            name = '컴퓨터'
        elif iconId == 834:
            name = '영화'
        elif iconId == 843:
            name = '돋보기'
        elif iconId == 856:
            name = '책'
        elif iconId == 935:
            name = '양궁'
        elif iconId == 937:
            name = '톱'
        elif iconId == 959:
            name = '피'
        newIcons.append([newId, icon['font'], name, icon['category']])
        newId += 1


df = pd.DataFrame(newIcons, columns=['iconId', 'font', 'name', 'category'])
df['name'] = df['name'].str.replace(" ", "", regex=False)
df.to_csv('newIcon.csv', index=False, encoding='utf-8')

sqlFile = open('newIcon_sql.txt', 'w', encoding='utf-8')
for (iconId, font, name, category) in newIcons:
    query = "INSERT INTO Icon(iconId, font, name, category) VALUES (" + str(iconId) + ", '" + font + "', '" + name + "', '" + category + "');\n"
    sqlFile.write(query)
sqlFile.close()