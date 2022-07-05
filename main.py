import math
import random
from matplotlib import pyplot as plt
from tkinter import *


class SolveTSPUsingACO:
    class Edge:     # класс феромонов
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:   #   класс для муравьишек
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def _select_node(self):    # правила по которым выбираем вершини
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, mode='', colony_size=0, elitist_weight=1.0, min_scaling_factor=0.01, alpha=0.0, beta=0.0,
                 p=0.0, pheromone_deposit_weight=0.0, initial_pheromone=0.2, steps=0, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.p = p
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)), initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _elitist(self, line_width=0.5, point_radius=3, annotation_size=8, dpi=120, save=True, name=None):
        plt.ion()  # Ключевая функция для успешного интерактивного режима
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.p)

            fig = plt.figure(1)

            x = [self.nodes[i][0] for i in self.global_best_tour]
            x.append(x[0])
            y = [self.nodes[i][1] for i in self.global_best_tour]
            y.append(y[0])
            plt.clf()
            plt.plot(x, y, linewidth=line_width)
            plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
            plt.title(self.mode)
            for i in self.global_best_tour:
                plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)

            plt.pause(0.05)  # Пауза на 0,05 секунды
            plt.ioff()

    def run(self):
        print('Started : {0}'.format(self.mode))
        if self.mode == 'Elitist':
            self._elitist()
        print('Ended : {0}'.format(self.mode))
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))

    def plot(self, line_width=0.5, point_radius=3, annotation_size=8, dpi=120, save=True, name=None):  #координатная плоскость
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
        plt.title(self.mode)
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
        # if save:    # save image by our way
        #     if name is None:
        #         name = '{0}.png'.format(self.mode)
        #     plt.savefig(name, dpi=dpi)
        plt.show()
        plt.gcf().clear()

def data():
    global _colony_size
    global _steps
    global _alpha   # коэф альфа
    global _beta    #  коэф бета
    global _p      # коэф испарения феромона
    global _pheromone_deposit_weight #   коэф Q

    if (Get_colony_size.get() == ""):
        _colony_size = 40
    else:
        _colony_size = int(Get_colony_size.get())
    if (Get_steps.get() == ""):
        _steps = 50
    else:
        _steps = int(Get_steps.get())
    if (Get_alpha.get() == ""):
        _alpha = 0.8
    else:
        _alpha = float(Get_alpha.get())
    if (Get_beta.get() == ""):
        _beta = 4
    else:
        _beta = float(Get_beta.get())
    if (Get_p.get() == ""):
        _p = 0.9
    else:
        _p = float(Get_p.get())
    if (Get_pheromone_deposit_weight.get() == ""):
        _pheromone_deposit_weight = 10.0
    else:
        _pheromone_deposit_weight = float(Get_pheromone_deposit_weight.get())

def wind_destroy():
    tk.destroy()

if __name__ == '__main__':
    tk = Tk()
    tk.title("Data")


    # _colony_size = 20  #количество муравишек
    # _steps = 100   #количество итераций
    _nodes = []   #наши вершинки
    # _alpha = 0.5  # коэф альфа
    # _beta = 7   #  коэф бета
    # _p = 0.5       # коэф испарения феромона
    # _pheromone_deposit_weight = 2.0   #   коэф Q

    label_colony_size = Label(tk, text="Кількість мурашок в колонії", font='Arial 15')
    label_colony_size.grid(row=1, column=1, sticky="w")
    Get_colony_size = Entry(tk)
    Get_colony_size.grid(row=1, column=2, sticky="w")
    label_steps = Label(tk, text="Кількість кроків", font='Arial 15')
    label_steps.grid(row=2, column=1, sticky="w")
    Get_steps = Entry(tk)
    Get_steps.grid(row=2, column=2, sticky="w")
    label_alpha = Label(tk, text="Коефіцієнт альфа", font='Arial 15')
    label_alpha.grid(row=3, column=1, sticky="w")
    Get_alpha = Entry(tk)
    Get_alpha.grid(row=3, column=2, sticky="w")
    label_beta = Label(tk, text="Коефіцієнт бета", font='Arial 15')
    label_beta.grid(row=4, column=1, sticky="w")
    Get_beta = Entry(tk)
    Get_beta.grid(row=4, column=2, sticky="w")
    label_p = Label(tk, text="Коефіцієнт випаровування феромону", font='Arial 15')
    label_p.grid(row=5, column=1, sticky="w")
    Get_p = Entry(tk)
    Get_p.grid(row=5, column=2, sticky="w")
    label_pheromone_deposit_weight = Label(tk, text="Ваговий коефіцієнт", font='Arial 15')
    label_pheromone_deposit_weight.grid(row=6, column=1, sticky="w")
    Get_pheromone_deposit_weight = Entry(tk)
    Get_pheromone_deposit_weight.grid(row=6, column=2, sticky="w")

    btn = Button(tk, text="Прийняти", command=lambda: [data(), wind_destroy()])
    btn.grid(row=7, column=2)
    tk.mainloop()

    with open('./att48.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            _nodes.append((int(city[1]), int(city[2])))

    elitist = SolveTSPUsingACO(mode='Elitist', colony_size=_colony_size, steps=_steps, nodes=_nodes, alpha=_alpha, beta=_beta, p=_p, pheromone_deposit_weight= _pheromone_deposit_weight)
    elitist.run()
    elitist.plot()