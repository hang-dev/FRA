import math
import numpy as np

np.random.seed(12345)

# STEP SIZE
delta = 1
# nSamples = 1000
L = 4



def U(x,subcircuit_entry_distributions,nA,nB):

    if isinstance(x,str) == False:
        # x = bin(math.fabs(math.floor(x)))[2:]
        x = bin(int(math.fabs(math.floor(x))))[2:]
        n = nA + nB
        x = x.zfill(n)

    indB = int(x[0: nB], 2)
    indA = int(x[nB:], 2)
    p = 0
    for i in range(2):
        subcircuit0_entry_distributions = subcircuit_entry_distributions[0]
        subcircuit1_entry_distributions = subcircuit_entry_distributions[1]
    for i in range(4):
        p += subcircuit0_entry_distributions[i][indA] * subcircuit1_entry_distributions[i][indB]
    p = p / 2
    p=math.fabs(p)
    if p != 0:
        p = -math.log(p)
    else:
        p = p

    return p



# DEFINE GRADIENT OF POTENTIAL ENERGY
def dU(U,x, subcircuit_entry_distributions, nA, nB, h=1e-6):

    if isinstance(x, str):
        x=int(x,2)


    return 2*(U(x + h, subcircuit_entry_distributions, nA, nB) - U(x - h, subcircuit_entry_distributions, nA, nB)) / 2 * h

# DEFINE KINETIC ENERGY FUNCTION
def K(p):

    return np.sum((p*p))/2


def reconstruct_approximate_HMC(subcircuit_entry_distributions,cuts, s=10000):
    # INITIAL STATE
    # print("_____")
    burn = int(0.4 * s)
    nA=cuts["subcircuits"][0].num_qubits-1
    nB=cuts["subcircuits"][1].num_qubits
    n=nA+nB
    p_rec = {}
    x = bin(np.random.randint(0,2**n))
    x = x[2:len(x)].zfill(n)

    for i in range(s + burn):
        # SAMPLE RANDOM MOMENTUM

        p0 = np.random.randn()
        grad_U = dU(U, x, subcircuit_entry_distributions, nA, nB)
        # FULL STEPS

        xStar = x
        pStar = p0
        for jL in range(L - 1):
            pStar = pStar - delta * grad_U / 2
            d = int(math.fabs(round(delta * pStar)))
            if d == n - 1:
                xStar = x[0:d] + str((int(xStar[d]) + 1) % 2)
            elif d >= n:
                d = np.random.randint(0,n)
                xStar = x[0:d] + str((int(xStar[d]) + 1) % 2) + x[d + 1:n]
            else:
                xStar = x[0:d] + str((int(xStar[d]) + 1) % 2) + x[d + 1:n]
            grad_U = dU(U, xStar, subcircuit_entry_distributions, nA, nB)
            pStar = pStar - delta * grad_U / 2


        U0 = U(x, subcircuit_entry_distributions, nA, nB)
        UStar = U(xStar, subcircuit_entry_distributions, nA, nB)

        K0 = K(p0)
        KStar = K(pStar)

        # ACCEPTANCE/REJECTION CRITERION
        alpha = min(1, np.exp((U0 + K0) - (UStar + KStar)))

        u = np.random.rand()

        if u < alpha:
            if isinstance(xStar,str):
                x = xStar
            else:
                x = xStar
                x = bin(int(math.fabs(math.floor(x))))[2:]
                x = x.zfill(n)

        if i > burn:
            if x not in p_rec:
                p_rec[x] = 1
            else:
                p_rec[x] += 1
    p_rec1 = {}
    sum_v = 0
    for value in p_rec.values():
        sum_v += value
    for k, v in p_rec.items():
        binary = k
        prob = v / sum_v
        p_rec1[binary] = prob

    return p_rec1