from tkinter import W
import numpy as np
import math
import seaborn as sns
import csv
import sklearn.metrics
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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

#test funkcija

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

#funkcija za ugibanje naslednjega vnosa

def ugib(podatki, vrstica, steviloSosedov):
    zbirkaSosedi = sosedi(podatki, vrstica, steviloSosedov)

    izpisi = [row[-1] for row in zbirkaSosedi]

    return max(set(izpisi), key=izpisi.count)

#funkcija knn

def knn(podatki, testni_podatki, steviloSosedov):
    ugibanja = list()

    for vrstica in testni_podatki:
        ugibanja.append(ugib(podatki, vrstica, steviloSosedov))

    return ugibanja



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


with open('bankovci.csv', newline='') as datoteka:
    vnosi = list(csv.reader(datoteka))

print(sosedi(dataset, dataset[5], 3))
print(ugib(dataset, dataset[5], 3))

print(list(razbitje(dataset, 3)))

print("_____________________" + '\n')

print(cross_validation_split(dataset, 3))