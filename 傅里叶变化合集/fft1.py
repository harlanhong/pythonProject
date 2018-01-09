import numpy as np
import math
if __name__ == '__main__':
    data = [15, 32, 9, 222, 18, 151, 5, 7, 56, 233, 56, 121, 235, 89, 98, 111]
    M = data.__len__()
    Cu = 1
    fft = []
    for u in range(M):
        sum = complex(0, 0)
        for x in range(M):
            temp = -2*math.pi*u*x/M
            sum+=(data[x]*(math.cos(temp)+1j*math.sin(temp)))
        fft.append(sum)
    print(fft)

    infft = []
    for x in range(M):
        sum = complex(0, 0)
        for u in range(M):
            power = 2*math.pi*u*x/M
            sum +=fft[u]*(math.cos(power)+1j*math.sin(power))
        infft.append(1/M*sum);
    print(infft);

