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

    return int(max(set(izpisi), key=izpisi.count))

#funkcija knn

def knn(podatki, testni_podatki, steviloSosedov):
    ugibanja = list()

    for vrstica in testni_podatki:
        ugibanja.append(ugib(podatki, vrstica, steviloSosedov))

    return ugibanja

#test funkcija

def accuracy_metric(actual, predicted):
    print("<<<<fold>>>>>")
    foldi = []

  
    akjurasi = accuracy_score(actual, predicted)
    print("Accuracy: ")
    print(akjurasi)
    tocnost.append(akjurasi)
    foldi.append(akjurasi)

    precizija = precision_score(actual, predicted, average='macro')
    print("Precision: ")
    print(precizija)
    preciznost.append(precizija)
    foldi.append(precizija)


    recall = recall_score(actual, predicted, average='macro')
    print("Recall: ")

    print(recall)
    priklic.append(recall)
    foldi.append(recall)
    
    obcutljivost = recall_score(actual, predicted, average='macro')
    print("Sensibility : ")
    print(obcutljivost)
    senzitivnost.append(obcutljivost)
    foldi.append(obcutljivost)


    f = f1_score(actual, predicted, average='macro')
    print("Fmera matrix: ")
    print(f)
    Fmera.append(f)
    foldi.append(f)

   
    print("Confusion matrix: ")

    
    print(confusion_matrix(actual, predicted))
    cm1 = confusion_matrix(actual, predicted)
    print("Specificnost")
    specificity1 = cm1[1, 1]/(cm1[1, 0]+cm1[1, 1])
    print(specificity1)
    specificnost.append(specificity1)
    foldi.append(specificity1)
    return foldi

def evaluate_algorithm(podatki, funkcija, steviloDelov, *args):
    deli = cross_validation_split(podatki, steviloDelov)
    rezultati = list()

    for delcek in deli:
        train_set = list(deli)
        train_set.remove(delcek)
        train_set = sum(train_set, [])
        test_set = list()
        for vrstica in delcek:
            row_copy = list(vrstica)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = funkcija(train_set, test_set, *args)
        actual = [row[-1] for row in delcek]
        accuracy = accuracy_metric(actual, predicted)
        rezultati.append(accuracy)
    
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

print(cross_validation_split(dataset, 3))

rezultati = evaluate_algorithm(podatki, knn, steviloDelov, num_neighbors)

print("Povrečna vrednost: ")
print('Povrečna vrednost točnost: %.3f%%' % (sum(tocnost)/float(len(tocnost))))
print('Povrečna vrednost priklic: %.3f%%' % (sum(priklic)/float(len(priklic))))
print('Povrečna vrednost preciznost: %.3f%%' %(sum(preciznost)/float(len(preciznost))))
print('Povrečna vrednost fmera: %.3f%%' % (sum(Fmera)/float(len(Fmera))))
print('Povrečna vrednost senzitivnost: %.3f%%' %(sum(senzitivnost)/float(len(senzitivnost))))
print('Povrečna vrednost specifičnost: %.3f%%' % (sum(specificnost)/float(len(specificnost))))
