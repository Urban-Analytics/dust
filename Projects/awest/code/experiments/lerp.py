import time
import numpy as np

def rand():
    loc1 = np.random.rand(2)
    loc2 = np.random.rand(2)
    speed = np.random.rand()
    return loc1, loc2, speed

# Lerps
def lerp0(loc1, loc2, speed):
    # Frobenius Norm  (standard matrix norm, p=2-norm)
    distance = np.linalg.norm(loc1 - loc2)
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

def lerp1(loc1, loc2, speed):
    # Euclidean Norm  (standard vector norm, p=2-norm)
    distance = np.sqrt(sum((loc1 - loc2)**2))
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

def lerp2(loc1, loc2, speed):
    # Manhattan Norm  (semi-standard vector norm. p=1-norm)
    distance = sum(abs(loc1 - loc2))
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

sqrt2 = np.sqrt(2)
def lerp3(loc1, loc2, speed):
    # Adjusted Manhattan Norm
    distance = sum(abs(loc1 - loc2)) / sqrt2
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

#sqrt2 = np.sqrt(2)
def lerp4(loc1, loc2, speed):
    # Adjusted Manhattan Norm with abbreviation
    loc = loc2 + speed * (loc1 - loc2) * sqrt2 / sum(abs(loc1 - loc2))
    return loc

#sqrt2 = np.sqrt(2)
def lerp5(loc1, loc2, speed):
    # Reciprocal of Adjusted Manhattan Norm
    reciprocal_distance = sqrt2 / sum(abs(loc1 - loc2))
    loc = loc2 + speed * (loc1 - loc2) * reciprocal_distance
    return loc


# Lerps Profiling
if 1:
    n = int(1e6)

    t = time.time()
    [lerp0(*rand()) for _ in range(n)]
    t = time.time() - t
    print(t/n)  # 8.050957202911376e-06

    t = time.time()
    [lerp1(*rand()) for _ in range(n)]
    t = time.time() - t
    print(t/n)  # 7.817470788955688e-06
    t = time.time()
    [lerp2(*rand()) for _ in range(n)]
    t = time.time() - t
    print(t/n)  # 6.1941430568695065e-06

    t = time.time()
    [lerp3(*rand()) for _ in range(n)]
    t = time.time() - t
    print(t/n)  # 6.3906128406524655e-06

    t = time.time()
    [lerp4(*rand()) for _ in range(n)]
    t = time.time() - t
    print(t/n)  # 7.330006837844849e-06

    t = time.time()
    [lerp5(*rand()) for _ in range(n)]
    t = time.time() - t
    print(t/n)  # 6.407449245452881e-06

# Manhattan Adjustment
if 1:
    sqrt2 = np.sqrt(2)
    import matplotlib.pyplot as plt
    I = int(1e5)
    high, low = 100, -100
    x = (high-low) * np.random.random(size=(I, 2)) - low
    ds = []
    for i in range(len(x)-1):
        x0 = x[i]
        x1 = x[i+1]
        n1 = sum(abs(x0-x1))           # mathattan norm
        n1adj = n1 / sqrt2             # adjusted manhattan norm
        n2 = np.sqrt(sum((x0-x1)**2))  # euclidean norm
        d = n1adj - n2
        ds.append(d)
    std = .6827  # 1 standard deviation
    prop = sum(np.array(ds) < std/2 * -(high-low)/2) / I
    perc = 100*(1-prop)
    plt.hist(ds)
    plt.title('Demonstrates Adjusted Manhattan is a lower bound\n{:.1f}% fall within 1 standard deviation\n\nEuclidean vs Adjusted-Manhattan Norm Histogram'.format(perc))
    plt.xlabel('Method difference (n1adj - n2)')
    plt.ylabel('Samples {:,}'.format(I))
    plt.show()
