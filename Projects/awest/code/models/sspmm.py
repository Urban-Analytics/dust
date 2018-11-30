# StationSim (pronounced Mike's Model)
'''
A interacting agent based model.
'''
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, model, unique_id):
        # Required
        self.unique_id = unique_id
        self.active = 0  # 0 Not Started, 1 Active, 2 Finished
        model.pop_active += 1
        # Location
        self.location = model.loc_entrances[np.random.randint(model.entrances)]
        self.location[1] += model.entrance_space * (np.random.uniform() - .5)
        self.loc_desire = model.loc_exits[np.random.randint(model.exits)]
        # Parameters
        self.time_activate = np.random.exponential(model.entrance_speed)
        self.speed_desire = max(np.random.normal(model.speed_desire_mean, model.speed_desire_std), 2*model.speed_min)
        self.speeds = np.arange(self.speed_desire, model.speed_min, -model.speed_step)
        if model.do_save:
            self.history_loc = []
        return

    def step(self, model):
        if self.active == 0:
            self.activate(model)
        elif self.active == 1:
            self.move(model)
            self.exit_query(model)
            self.save(model)
        return

    def activate(self, model):
        if not self.active and model.time_id > self.time_activate:
            self.active = 1
            self.time_start = model.time_id
            self.time_expected = np.linalg.norm(self.location - self.loc_desire) / self.speed_desire
        return

    def move(self, model):
        for speed in self.speeds:
            # Direct
            new_location = self.lerp(self.loc_desire, self.location, speed)
            if not self.collision(model, new_location):
                break
            elif speed == self.speeds[-1]:
                # Wiggle
                new_location = self.location + np.random.randint(-1, 1+1, 2)
        # Rebound
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
        # Move
        self.location = new_location
        return

    def collision(self, model, new_location):
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            collide = True
        elif self.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    def neighbourhood(self, model, new_location, do_kd_tree=True):
        neighbours = False
        neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
        for neighbouring_agent in neighbouring_agents:
            agent = model.agents[neighbouring_agent]
            if agent.active == 1 and new_location[0] <= agent.location[0]:
                neighbours = True
                break
        return neighbours

    def lerp(self, loc1, loc2, speed):
        distance = np.linalg.norm(loc1 - loc2)
        loc = loc2 + speed * (loc1 - loc2) / distance
        return loc

    def exit_query(self, model):
        if np.linalg.norm(self.location - self.loc_desire) < model.exit_space:
            self.active = 2
            model.pop_active -= 1
            model.pop_finished += 1
            if model.do_save:
                time_delta = model.time_id - self.time_start
                model.time_taken.append(time_delta)
                time_delta -= self.time_expected
                model.time_delayed.append(time_delta)
        return

    def save(self, model):
        if model.do_save:
            self.history_loc.append(self.location)
        return


class Model:

    def __init__(self, params):
        self.params = (params,)
        [setattr(self, key, value) for key, value in params.items()]
        self.speed_step = (self.speed_desire_mean - self.speed_min) / 3  # Average number of speeds to check
        # Batch Details
        self.time_id = 0
        self.step_id = 0
        if self.do_save:
            self.time_taken = []
            self.time_delayed = []
        # Model Parameters
        self.boundaries = np.array([[0, 0], [self.width, self.height]])
        self.pop_active = 0
        self.pop_finished = 0
        # Initialise
        self.initialise_gates()
        self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
        return

    def step(self):
        if self.pop_finished < self.pop_total and self.step:
            self.kdtree_build()
            [agent.step(self) for agent in self.agents]
        self.time_id += 1
        self.step_id += 1
        return

    def initialise_gates(self):
        # Entrances
        self.loc_entrances = np.zeros((self.entrances, 2))
        self.loc_entrances[:, 0] = 0
        if self.entrances == 1:
            self.loc_entrances[0, 1] = self.height / 2
        else:
            self.loc_entrances[:, 1] = np.linspace(self.height / 4, 3 * self.height / 4, self.entrances)
        # Exits
        self.loc_exits = np.zeros((self.exits, 2))
        self.loc_exits[:, 0] = self.width
        if self.exits == 1:
            self.loc_exits[0, 1] = self.height / 2
        else:
            self.loc_exits[:, 1] = np.linspace(self.height / 4, 3 * self.height / 4, self.exits)
        return

    def kdtree_build(self):
        state = self.agents2state(do_ravel=False)
        self.tree = cKDTree(state)
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
            self.agents[i].location = state[2 * i:2 * i + 2]
        return

    def batch(self):
        for i in range(self.batch_iterations):
            self.step()
            if self.do_ani:
                self.ani()
            if self.pop_finished == self.pop_total:
                print('Everyone made it!')
                break
        if self.do_save:
            self.save_stats()
            self.save_plot()
        return

    def ani(self):
        plt.figure(1)
        plt.clf()
        for agent in self.agents:
            if agent.active == 1:
                plt.plot(*agent.location, '.k')#, markersize=4)
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.xlabel('Corridor Width')
        plt.ylabel('Corridor Height')
        plt.pause(1 / 30)
        return

    def save_ani(self):



        return

    def save_plot(self):
        # Trails
        plt.figure()
        for agent in self.agents:
            if agent.active == 0:
                colour = 'r'
            elif agent.active == 1:
                colour = 'b'
            else:
                colour = 'm'
            locs = np.array(agent.history_loc).T
            plt.plot(locs[0], locs[1], color=colour, linewidth=.5)
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.xlabel('Corridor Width')
        plt.ylabel('Corridor Height')
        plt.legend(['Agent trails'])
        # Time Taken, Delay Amount
        plt.figure()
        plt.hist(self.time_taken, alpha=.5, label='Time taken')
        plt.hist(self.time_delayed, alpha=.5, label='Time delay')
        plt.xlabel('Time')
        plt.ylabel('Number of Agents')
        plt.legend()

        plt.show()
        return

    def save_stats(self):
        print()
        print('Stats:')
        print('Finish Time: ' + str(self.time_id))
        print('Active / Finished / Total agents: ' + str(self.pop_active) + '/' + str(self.pop_finished) + '/' + str(self.pop_total))
        print('Average time taken: ' + str(np.mean(self.time_taken)) + 's')
        return


def easy_model():
    model_params = {
        'width': 200,
        'height': 100,
        'pop_total': 100,
        'entrances': 3,
        'entrance_space': 2,
        'entrance_speed': .1,
        'exits': 2,
        'exit_space': 1,
        'speed_min': .1,
        'speed_desire_mean': 1,
        'speed_desire_std': 1,
        'separation': 2,
        'batch_iterations': 400,
        'do_save': False,
        'do_ani': True,
        }
    model = Model(model_params)
    return model


if __name__ == '__main__':
    model = easy_model()
    model.batch()
