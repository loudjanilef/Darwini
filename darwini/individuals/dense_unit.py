import random

from keras.layers.core import Dense, Dropout
from keras.models import Sequential

import darwini.constants as constants
from darwini.individuals.individual_unit import IndividualUnit


class DenseUnit(IndividualUnit):
    size: int
    activation: str
    dropout_rate: float

    def __init__(self, size: int, activation: str, dropout_rate: float) -> None:
        self.size = size
        self.activation = activation
        self.dropout_rate = dropout_rate

    @staticmethod
    def generate() -> 'DenseUnit':
        size = random.randint(constants.MIN_DENSE_SIZE, constants.MAX_DENSE_SIZE)
        activation = random.choice(constants.ACTIVATIONS)
        dropout_rate = random.gauss(0.5, 0.2)
        return DenseUnit(size, activation, dropout_rate)

    def blend(self, partner: 'DenseUnit') -> 'DenseUnit':
        size = random.choice([self.size, partner.size])
        activation = random.choice([self.activation, partner.activation])
        dropout_rate = random.choice([self.dropout_rate, partner.dropout_rate])
        return DenseUnit(size, activation, dropout_rate)

    def mutate(self) -> 'DenseUnit':
        size = self.size
        activation = self.activation
        dropout_rate = self.dropout_rate
        if random.random() < constants.MUTATION_RATE:
            size += random.gauss(0, 0.2)
            activation = random.choice(constants.ACTIVATIONS)
            dropout_rate += random.gauss(0, 0.2)
        return DenseUnit(size, activation, dropout_rate)

    def add_to_network(self, network: Sequential) -> None:
        network.add(Dense(self.size, activation=self.activation))
        network.add(Dropout(self.dropout_rate))

    def __eq__(self, o: 'DenseUnit') -> bool:
        if type(self) != type(o):
            return False

        return self.size == o.size and self.activation == o.activation and self.dropout_rate == o.dropout_rate
