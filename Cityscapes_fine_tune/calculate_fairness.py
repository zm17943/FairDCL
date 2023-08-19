import numpy as np
import os

top_city = ['hamburg', 'cologne', 'dusseldorf', 'hanover', 'stuttgart', 'frankfurt', 'bremen']
med_city = ['aachen', 'ulm', 'munster', 'tubingen', 'bochum', 'krefeld']
low_city = ['darmstadt', 'jena', 'monchengladbach', 'strasbourg', 'weimar', 'erfurt', 'lindau']

a = np.load('IoU_results5k.npy', allow_pickle=True).item()

top_mean = 0

for city in top_city:
    top_mean += np.array(a[city]).mean()

top_mean /= len(top_city)


med_mean = 0
for city in med_city:
    med_mean += np.array(a[city]).mean()

med_mean /= len(med_city)


low_mean = 0
for city in low_city:
    low_mean += np.array(a[city]).mean()

low_mean /= len(low_city)


print(top_mean)
print(med_mean)
print(low_mean)

print((top_mean+med_mean+low_mean)/3)
ss = top_mean+med_mean+low_mean
print((abs(top_mean/ss-1/3) + abs(med_mean/ss-1/3) + abs(low_mean/ss-1/3)) * 1.5)

