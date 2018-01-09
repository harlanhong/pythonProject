import math
import numpy as np
if __name__ == '__main__':
    data=[[1,1,3,2],[3,4,123,154],[55,2,22,233]]
    fft2 = data.copy()
    M = 4
    N = 3

    for u in range(N):
        for v in range(M):
            fft2[u][v] = 0+0j
            for x in range(N):
                for y in range(M):
                    power = -2*math.pi*(u*x/N+v*y/M)
                    fft2[u][v] +=(data[x][y]*(math.cos(power)+1j*math.sin(power)))

    print(fft2)
