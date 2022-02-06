import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plot
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

    for podatek in podatki:
        dolzina = evklid(vrstica, podatek)
        dolzine[stevec] = dolzina
        sosedi.append((podatek, dolzina))
        stevec += 1

    sosedi.sort(key=lambda parameter: parameter[1])

    for i in range(steviloSosedov):
        praviSosedi.append(sosedi[i][0])

    return praviSosedi

#funkcija za razbitje podatkov na manjše dele

def razbitje(podatki, steviloDelov):
    tabela = np.array(podatki)
    np.random.shuffle(tabela)

    noviPodatki = np.array_split(tabela, steviloDelov)

    return noviPodatki

#funkcija za ugibanje naslednjega vnosa

def ugib(podatki, vrstica, steviloSosedov):
    zbirkaSosedi = sosedi(podatki, vrstica, steviloSosedov)

    izpisi = [sosed[-1] for sosed in zbirkaSosedi]

    return int(max(set(izpisi), key=izpisi.count))

#funkcija knn

def knn(podatki, testni_podatki, steviloSosedov):
    ugibanja = list()

    for vrstica in testni_podatki:
        ugibanja.append(ugib(podatki, vrstica, steviloSosedov))

    return ugibanja

#funkcija za izracun natancnosti

def natancnostFolda(aktualno, napoved, stevec):
    precizija = precision_score(aktualno, napoved, average='macro')
    preciznost.append(precizija)

    akjurasi = accuracy_score(aktualno, napoved)
    tocnost.append(akjurasi)

    f = f1_score(aktualno, napoved, average='macro')
    Fmera.append(f)

    priklich = recall_score(aktualno, napoved, average='macro')
    priklic.append(priklich)
    
    matrikaZmeda = confusion_matrix(aktualno, napoved)

    obcutljivost = recall_score(aktualno, napoved, average='macro')
    senzitivnost.append(obcutljivost)
   

    specMatrika = matrikaZmeda[1, 1]/(matrikaZmeda[1, 0]+matrikaZmeda[1, 1])
    specificnost.append(specMatrika)

    print("~> Fold stevilka " + str(stevec) + " <~")
    print('\n')
    print('\n')
    print("Tocnost: " + str(precizija))
    print("Natancnost: " + str(akjurasi))
    print("Priklic: " + str(priklich))
    print("F-mera: " + str(f))
    print("Obcutljivost: " + str(obcutljivost))
    print("Matrika zmede: ")
    print(matrikaZmeda)
    print("Specificnost: " + str(specMatrika))
    print('\n')


# koncni algoritem

def algoritem(podatki, funkcija, steviloDelov, *args):
    deli = razbitje(podatki, steviloDelov)
    stevec = 0

    for delcek in deli:
        trening = np.array(deli, dtype=object)
        trening = np.delete(trening, stevec)
        trening = np.concatenate(trening)
        test = list()

        for vrstica in delcek:
            vrstica2 = list(vrstica)
            test.append(vrstica2)
            vrstica2[-1] = None

        napoved = funkcija(trening, test, *args)
        aktualno = [row[-1] for row in delcek]
        natancnostFolda(aktualno, napoved, stevec + 1)
        stevec += 1
    

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

steviloSosedov = 5
steviloDelov = 5

with open('bankovci.csv', newline='') as datoteka:
    vnosi = list(csv.reader(datoteka))
    podatki = list()
    vnosi.pop(0)

    for vrstica in vnosi:
        sprememba = [float(vrstica[0]), float(vrstica[1]), float(vrstica[2]), float(vrstica[3]), int(vrstica[4])]
        podatki.append(sprememba)

algoritem(podatki, knn, steviloDelov, steviloSosedov)

print('\n')
print("~> REZULTATI <~")
print('Povrecna vrednost priklic -> ' + str(np.average(np.array(priklic))))
print('Povrecna vrednost priklic -> ' + str(np.average(np.array(priklic))))
print('Povrecna vrednost preciznost -> ' + str(np.average(np.array(preciznost))))
print('Povrecna vrednost fmera -> ' + str(np.average(np.array(Fmera))))
print('Povrecna vrednost senzitivnost -> ' + str(np.average(np.array(senzitivnost))))
print('Povrecna vrednost specificnost -> ' + str(np.average(np.array(specificnost))))