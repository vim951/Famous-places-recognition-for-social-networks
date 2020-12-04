##

import pandas as pd
import numpy as np
import httplib2

##

def isDead(url):
    h = httplib2.Http()
    resp = h.request(url, 'HEAD')
    return not (int(resp[0]['status'])<400)

c_step=0
def showProgress(step, total, nb_step):
    global c_step
    if step/total > c_step/nb_step:
        c_step+=1
        print("Progress: " + str(round(100*100*step/total)/100) + "%")
        print("Current approximation of dead links: " + str(round(100*100*d/step)/100) + "%")
        print("Error approximation: " + str(round(100*100*e/step)/step) + "%")
        print("="*c_step + "-"*(nb_step-c_step))

##

df = pd.read_csv('train.csv')
values = df.values
n = len(values)
d=0
e=0

for i in range(n):
    if i>0:
        showProgress(i, n, 100)
    try:
        d+= 1 if isDead(values[i][1]) else 0
    except:
        e+=1

print("Errors: " + str(round(100*100*d/n)/100) + "%")
print("Dead links: " + str(round(100*100*d/n)/100) + "%")