import numpy as np

# MinMaxScaler
def My_MinMaxScaler(data,minmaxrange = [None , None] ,round=3):
    data = np.array(data)
    if minmaxrange[0] == None:
        xmin = data.min()

    elif minmaxrange[1] == None:
        xmax = data.max()

    else :
        xmin = minmaxrange[0]
        xmax = minmaxrange[1]
    newData = []
    for i in range(len(data)):
        for x in range(i):
            x = (x - xmin)/(xmax - xmin)
            newData.append(x.round(round))

    return newData
