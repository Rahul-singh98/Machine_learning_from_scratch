from utils import My_MinMaxScaler

data = [[1,2,3,4,5,6,8,9,7,10,11,12,13,14,15,16,17,18,19,20],
        [1,5,2,7,3,2,4,6,5,34,32,43,23,44,23,43,43,64,22,32]]

scaledData =My_MinMaxScaler(data ,round = 5 , minmaxrange=[0,1])
print(scaledData)
