
# Will produce errors if imported
#    from multiprocessing import Pool
# Works if imported
from multiprocessing.dummy import Pool

if 0:
    import numpy as np

    class Model:
        def __init__(self, unique_id):
            self.unique_id = unique_id
            return
        def step(self):
            print(self.unique_id)
            return

    models = list([Model(unique_id) for unique_id in range(10)])

    def f(p):
        models[p].step()
        return

    if __name__ == '__main__':
        with Pool(2) as pool:
            pool.map(f, np.arange(len(models)))


if 1:
    def beers(i):
        if i == 1:
            print("\n{0} green bottle sitting on the wall,\n{0} green bottle sitting on the wall,\nAnd if one green bottle should accidentally fall,\nThere’ll be no green bottles hanging on the wall.".format(i))
        else:
            print("\n{0} green bottles sitting on the wall,\n{0} green bottles sitting on the wall,\nAnd if one green bottle should accidentally fall,\nThere'll be {1} green bottles hanging on the wall.".format(i, i-1))
        return

    Pool(4).map(print, range(100, 0, -1))
    #list(map(beers, range(99, 0, -1)))


if 0:
    import matplotlib.pyplot as plt

    def f(x):
        y = complex(x**.6)
        return y

    x = range(-50_000, 50_000)
    y = Pool().map(f, x)

    plt.plot(x,y)
    plt.show()

