import numpy as np

string = 'JSD_seed1234/IoU_results1k.npy'
string = 'IoU_results5k.npy'

top_city = ['hamburg', 'cologne', 'dusseldorf', 'hanover', 'stuttgart', 'frankfurt', 'bremen']
med_city = ['aachen', 'ulm', 'munster', 'tubingen', 'bochum', 'krefeld', "zurich"]
low_city = ['darmstadt', 'jena', 'monchengladbach', 'strasbourg', 'weimar', 'erfurt', 'lindau']

def compute(a):
    top = []
    med = []
    low = []
    for city in top_city:
        top.append(np.array(a[city]).mean())
    for city in med_city:
        med.append(np.array(a[city]).mean())
    for city in low_city:
        low.append(np.array(a[city]).mean())
    mean1 = sum(top)/len(top)
    mean2 = sum(med)/len(med)
    mean3 = sum(low)/len(low)
    diff = abs(mean1/(mean1+mean2+mean3)-1/3) + abs(mean2/(mean1+mean2+mean3)-1/3) + abs(mean3/(mean1+mean2+mean3)-1/3)
    print([mean1, mean2, mean3, diff])


compute(np.load(string, allow_pickle=True).item())
