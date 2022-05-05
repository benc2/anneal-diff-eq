from numpy import arange, asarray
import matplotlib.pyplot  as plt

def basisFunctions(N,nodes = [], shape = "triangle"):
    if nodes == []:
        nodes = arange(0,N+1)/N
    phi_evaluation = lambda i, x:        (x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])  if(i> 0 and x >= nodes[i - 1]  and x < nodes[i] ) \
        else (1 - (x - nodes[i]) / (nodes[i + 1] - nodes[i])) if(i<N  and x >= nodes[i] and (x < nodes[i + 1]) ) \
        else 0
    def phi_output(j,xs):
        output = []
        for x in xs:
            output.append(phi_evaluation(j,x))

        output = asarray(output)
        return output

    return phi_output
N=10
phi = basisFunctions(N )
for j in range(N+1):
    plt.plot(arange(0,1,0.001), phi(j,arange(0,1,0.001)))
    print(phi(j,arange(0,5,0.1)))

plt.show()

