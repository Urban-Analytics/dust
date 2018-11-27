# Particle Filter
'''
TODO:
    Multiprocessing
    time -> step_id
'''
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from multiprocessing import Pool



class Agent:

    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.active = np.random.randint(2)
        self.location = np.random.uniform(size=2)
        return

    def step(self):
        self.move()
        return

    def move(self):
        self.location += .01
        self.location %= 1
        return


class Model:

    def __init__(self, population):
        self.params = (population,)
        self.boundaries = np.array([[0, 0], [1, 1]])
        self.agents = list([Agent(unique_id) for unique_id in range(population)])
        return

    def step(self):
        [agent.step() for agent in self.agents]
        return

    def agents2state(self, do_ravel=True):
        state = [agent.location for agent in self.agents]
        if do_ravel:
            state = np.ravel(state)
        else:
            state = np.array(state)
        return state

    def state2agents(self, state):
        for i in range(len(self.agents)):
            self.agents[i].location = state[2*i:2*i+2]
        return

    def ani(model):
        state = model.agents2state(do_ravel=False).T
        plt.clf()
        plt.plot(state[0], state[1], '.k', alpha=.5)
        plt.axis((0, 1, 0, 1))
        plt.pause(.1)
        return


class ParticleFilter:

    def __init__(self, model, filter_params):
        [setattr(self, key, value) for key, value in filter_params.items()]
        self.time = 0
        self.base_model = model
        self.models = list([deepcopy(model) for _ in range(self.number_of_particles)])
        if not self.do_copies:
            [model.__init__(*model.params) for model in self.models]
        self.dimensions = len(self.base_model.agents2state())
        self.states = np.zeros((self.number_of_particles, self.dimensions))
        self.weights = np.ones(self.number_of_particles)
        if self.do_save:
            self.active_agents = []
            self.means = []
            self.mean_errors = []
            self.variances = []
        self.states = np.array(pool.map(self.initial_state, range(self.number_of_particles)))
        return

    def step(self):
        self.predict()
        if not self.time % self.resample_window:
            self.reweight()
            self.resample()
        if self.do_save:
            self.save()
        return

    def predict(self):
        self.time += 1
        self.base_model.step()

        stepped_particles = pool.map(self.step_particles, range(self.number_of_particles))

        self.models = [stepped_particles[i][0] for i in range(len(stepped_particles))]
        self.states = np.array([stepped_particles[i][1] for i in range(len(stepped_particles))])
        return

    def initial_state(self, particle):
        self.states[particle] = self.base_model.agents2state()
        return self.states[particle]

    def assign_agents(self, particle):
        self.models[particle].state2agents(self.states[particle])
        return self.models[particle]

    def step_particles(self, particle):
        self.models[particle].state2agents(self.states[particle])
        self.models[particle].step()
        self.states[particle] = self.models[particle].agents2state()
        return self.models[particle], self.states[particle]

    def reweight(self):
        measured_state = self.base_model.agents2state()
        distance = np.linalg.norm(self.states - measured_state, axis=1)
        self.weights = 1 / (distance + 1e-99)
        self.weights /= np.sum(self.weights)
        return

    def resample(self):
        offset_partition = np.arange(self.number_of_particles)
        cumsum = np.cumsum(self.weights)
        i, j = 0, 0
        indexes = np.zeros(self.number_of_particles, 'i')
        while i < self.number_of_particles and j < self.number_of_particles:
            if offset_partition[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        self.states[:] = self.states[indexes]
        self.weights[:] = self.weights[indexes]

        self.models = pool.map(self.assign_agents, range(self.number_of_particles))
        return

    def save(self):

        self.active_agents.append(sum([agent.active == 1 for agent in self.base_model.agents]))

        mean = np.average(self.states, weights=self.weights, axis=0)
        variance = np.average((self.states - mean)**2, weights=self.weights, axis=0)

        self.means.append(mean[:])
        self.variances.append(np.average(variance))

        truth_state = self.base_model.agents2state()
        self.mean_errors.append(np.linalg.norm(mean - truth_state, axis=0))

        return

    def save_plot(self):
        plt.figure(2)
        plt.plot(self.active_agents)
        plt.ylabel('Active agents')
        plt.show()

        plt.figure(3)
        plt.plot(self.mean_errors)
        plt.ylabel('Mean Error')
        plt.show()

        plt.figure(4)
        plt.plot(self.variances)
        plt.ylabel('Mean Variance')
        plt.show()

        print('Max mean error = ', max(self.mean_errors))
        print('Average mean error = ', np.average(self.mean_errors))
        print('Max mean variance = ', max(self.variances))
        print('Average mean variance = ', np.average(self.variances))

    def ani(self, agents_to_visualise):

        if any([agent.active == 1 for agent in self.base_model.agents]):

            plt.figure(1)
            plt.clf()

            markersizes = self.weights
            if np.std(markersizes) != 0:
                markersizes *= 4 / np.std(markersizes)   # revar
            markersizes += 4 - np.mean(markersizes)  # remean

            particle = -1
            for model in self.models:
                particle += 1
                markersize = np.clip(markersizes[particle], .5, 8)
                for agent in model.agents[:agents_to_visualise]:
                    if agent.active == 1:
                        unique_id = agent.unique_id
                        if self.base_model.agents[unique_id].active == 1:
                            locs = np.array([self.base_model.agents[unique_id].location, agent.location]).T
                            plt.plot(*locs, '-k', alpha=.1, linewidth=.3)
                            plt.plot(*agent.location, '.r', alpha=.3, markersize=markersize)

            for agent in self.base_model.agents:
                if agent.active == 1:
                    plt.plot(*agent.location, '+k')

            plt.axis(np.ravel(self.base_model.boundaries, 'F'))
            plt.pause(1 / 4)
        return


if __name__ == '__main__':  # basic
    pool = Pool()

    model = Model(100)

    filter_params = {
        'number_of_particles': 10,
        'number_of_iterations': 200,
        'particle_std': 0,
        'resample_window': 1,
        'do_copies': False,
        'do_save': True
        }
    pf = ParticleFilter(model, filter_params)

    for _ in range(200):
        model.step()
        #true_state = model.agents2state()
        #measured_state = true_state + np.random.normal(0, 0., true_state.shape)
        pf.step()
        pf.ani(2)
    pf.save_plot()
    plt.show()
