from multiprocessing import Pool
import numpy as np

class Model:
    def __init__(self):
        self.loc = np.random.rand()
        return

    def step(self):
        print(self.loc)
        return

models = list([Model() for _ in range(10)])


def f(p):
    models[p].step()
    return


if __name__ == '__main__':

    with Pool(4) as pool:
        pool.map(f, np.arange(len(models)))
