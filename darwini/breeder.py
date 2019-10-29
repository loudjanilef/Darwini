import random
from itertools import combinations
from typing import List, Tuple

from keras import Sequential
from keras.callbacks import EarlyStopping
from numpy.core.records import ndarray

from darwini import constants
from darwini.individuals.network import Network


class Breeder:
    population_size: int = 100
    population: List[Tuple[int, Network, Sequential]]
    selected: List[Tuple[int, Network, Sequential]]
    train_x: ndarray
    train_y: ndarray
    val_x: ndarray
    val_y: ndarray
    generation_nbr: int = 0
    input_shape: List[int]
    output_shape: int

    def __init__(self, train_x: ndarray, train_y: ndarray, val_x: ndarray, val_y: ndarray) -> None:
        self.population = []
        self.selected = []
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.input_shape = train_x.shape[1:]
        self.output_shape = train_y.shape[-1]

    def __initialize(self):
        for i in range(self.population_size):
            success = False
            while not success:
                network = Network.generate(self.input_shape, self.output_shape)
                success = self.__compile_and_fit(network, i + 1)
        self.__select()
        return self.population[0][0], self.population[0][2]

    def generation(self):
        if self.generation_nbr == 0:
            self.generation_nbr += 1
            return self.__initialize()

        self.generation_nbr += 1
        self.population = self.selected
        for i, pair in enumerate(combinations(self.selected, 2)):
            success = False
            while not success:
                print("retry")
                network = pair[0][1].blend(pair[1][1])
                network.mutate()
                success = self.__compile_and_fit(network, i + 1)
        self.__select()
        return self.population[0][0], self.population[0][2]

    def __compile_and_fit(self, network, generation) -> bool:
        try:
            model = network.compile()
            early_stopper = EarlyStopping(patience=3)
            print("Generation {} : Training model {}/{}".format(self.generation_nbr, generation, self.population_size))
            model.fit(self.train_x, self.train_y, batch_size=constants.BATCH_SIZE, epochs=constants.EPOCH_NBR,
                      verbose=1, validation_data=(self.val_x, self.val_y), callbacks=[early_stopper])
        except KeyboardInterrupt:
            raise
        except:
            return False
        score = model.evaluate(self.val_x, self.val_y, verbose=0)
        self.population.append((score[1], network, model))
        return True

    def __select(self):
        self.population.sort(key=lambda item: item[0], reverse=True)
        self.selected = self.population[:5]
        self.selected.extend(random.sample(self.population[:5], 5))
        for i, select in enumerate(self.selected):
            select[1].save("gen{}elem{}".format(self.generation_nbr, i))
