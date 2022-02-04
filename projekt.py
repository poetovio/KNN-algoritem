from tkinter import W
import numpy as np
import math
import seaborn as sns
import csv
import sklearn.metrics

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

print(sosedi(dataset, dataset[0], 3))