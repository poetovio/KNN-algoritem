from tkinter import W
import numpy as np
import math
import seaborn as sns
import csv
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#tabele za hranjenje podatkov

tocnost = list()
senzitivnost = list()
specificnost = list()
Fmera = list()
zmeda = list()
preciznost = list()
priklic = list()

#funkcija za izračun evklidske razdalje

def evklid(vrstica1, vrstica2):
    razdalja = 0.0
    for i in range(len(vrstica1) - 1):
        razdalja += (vrstica1[i] - vrstica2[i])**2
    
    return math.sqrt(razdalja)

#funkcija za pridobivanje sosedov

def sosedi(podatki, vrstica, steviloSosedov):
    dolzine = np.zeros(len(podatki))
    sosedi = list()
    praviSosedi = list()
    stevec = 0

    for row in podatki:
        dolzina = evklid(vrstica, row)
        dolzine[stevec] = dolzina
        sosedi.append((row, dolzina))
        stevec += 1

    sosedi.sort(key=lambda parameter: parameter[1])

    for i in range(steviloSosedov):
        praviSosedi.append(sosedi[i][0])

    return praviSosedi

#funkcija za razbitje podatkov na manjše dele

def razbitje(podatki, steviloDelov):
    velikost = int(len(podatki) / steviloDelov)
    tabela = np.array(podatki)
    np.random.shuffle(tabela)

    noviPodatki = np.array_split(tabela, steviloDelov)

    return noviPodatki

#funkcija za ugibanje naslednjega vnosa

def ugib(podatki, vrstica, steviloSosedov):
    zbirkaSosedi = sosedi(podatki, vrstica, steviloSosedov)

    izpisi = [row[-1] for row in zbirkaSosedi]

    return int(max(set(izpisi), key=izpisi.count))

#funkcija knn

def knn(podatki, testni_podatki, steviloSosedov):
    ugibanja = list()

    for vrstica in testni_podatki:
        ugibanja.append(ugib(podatki, vrstica, steviloSosedov))

    return ugibanja

#test funkcija

def natancnostFolda(actual, predicted, stevec):
    tabela = []

    akjurasi = accuracy_score(actual, predicted)
    tocnost.append(akjurasi)
    tabela.append(akjurasi)

    precizija = precision_score(actual, predicted, average='macro')
    preciznost.append(precizija)
    tabela.append(precizija)


    recall = recall_score(actual, predicted, average='macro')

    priklic.append(recall)
    tabela.append(recall)
    
    obcutljivost = recall_score(actual, predicted, average='macro')
    senzitivnost.append(obcutljivost)
    tabela.append(obcutljivost)


    f = f1_score(actual, predicted, average='macro')
    Fmera.append(f)
    tabela.append(f)

   
    matrikaZmeda = confusion_matrix(actual, predicted)

    specMatrika = matrikaZmeda[1, 1]/(matrikaZmeda[1, 0]+matrikaZmeda[1, 1])
    specificnost.append(specMatrika)

    tabela.append(specMatrika)


    print("~> Fold stevilka " + str(stevec) + " <~")
    print('\n')
    print('\n')
    print("Natancnost: ")
    print(akjurasi)
    print('\n')
    print("Tocnost: ")
    print(precizija)
    print('\n')
    print("Recall: ")
    print(recall)
    print('\n')
    print("Obcutljivost: ")
    print(obcutljivost)
    print('\n')
    print("Matrika f-mere: ")
    print(f)
    print('\n')
    print("Matrika zmede: ")
    print(matrikaZmeda)
    print('\n')
    print("Specificnost:")
    print(specMatrika)
    print('\n')
    print('\n')

    return tabela

def algoritem(podatki, funkcija, steviloDelov, *args):
    deli = razbitje(podatki, steviloDelov)
    rezultati = list()
    stevec = 0

    for delcek in deli:
        trening = np.array(deli, dtype=object)
        trening = np.delete(trening, stevec)
        #train_set = sum(train_set, [])
        trening = np.concatenate(trening)
        test = list()

        for vrstica in delcek:
            vrstica2 = list(vrstica)
            test.append(vrstica2)
            vrstica2[-1] = None

        napoved = funkcija(trening, test, *args)
        aktualno = [row[-1] for row in delcek]
        natancnost = natancnostFolda(aktualno, napoved, stevec + 1)
        rezultati.append(natancnost)
        stevec += 1
    
    return rezultati

#test

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

num_neighbors = 5
steviloDelov = 5

with open('bankovci.csv', newline='') as datoteka:
    vnosi = list(csv.reader(datoteka))
    podatki = list()
    vnosi.pop(0)

    for vrstica in vnosi:
        sprememba = [float(vrstica[0]), float(vrstica[1]), float(vrstica[2]), float(vrstica[3]), int(vrstica[4])]
        podatki.append(sprememba)

rezultati = algoritem(podatki, knn, steviloDelov, num_neighbors)

print('\n')
print("Povrečna vrednost: ")
print('Povrečna vrednost točnost: %.3f%%' % (sum(tocnost)/float(len(tocnost))))
print('Povrečna vrednost priklic: %.3f%%' % (sum(priklic)/float(len(priklic))))
print('Povrečna vrednost preciznost: %.3f%%' %(sum(preciznost)/float(len(preciznost))))
print('Povrečna vrednost fmera: %.3f%%' % (sum(Fmera)/float(len(Fmera))))
print('Povrečna vrednost senzitivnost: %.3f%%' %(sum(senzitivnost)/float(len(senzitivnost))))
print('Povrečna vrednost specifičnost: %.3f%%' % (sum(specificnost)/float(len(specificnost))))