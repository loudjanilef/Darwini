import random

from keras.layers.core import Dense, Dropout
from keras.models import Sequential

import darwini.constants as constants
from darwini.individuals.individual_unit import IndividualUnit


class DenseUnit(IndividualUnit):
    size: int
    activation: str
    has_dropout: bool
    dropout_rate: float

    def __init__(self, size: int, activation: str, has_dropout: bool, dropout_rate: float) -> None:
        self.size = size
        self.activation = activation
        self.has_dropout = has_dropout
        self.dropout_rate = dropout_rate

    @staticmethod
    def generate() -> 'DenseUnit':
        size = random.randint(constants.MIN_DENSE_SIZE, constants.MAX_DENSE_SIZE)
        activation = random.choice(constants.ACTIVATIONS)
        has_dropout = random.random() < 0.5
        dropout_rate = min(max(random.gauss(0.25, 0.2), 0), 0.5)
        return DenseUnit(size, activation, has_dropout, dropout_rate)

    def blend(self, partner: 'DenseUnit') -> 'DenseUnit':
        size = random.choice([self.size, partner.size])
        activation = random.choice([self.activation, partner.activation])
        has_dropout = random.choice([self.has_dropout, partner.has_dropout])
        dropout_rate = random.choice([self.dropout_rate, partner.dropout_rate])
        return DenseUnit(size, activation, has_dropout, dropout_rate)

    def mutate(self) -> 'DenseUnit':
        size = self.size
        activation = self.activation
        has_dropout = self.has_dropout ^ (random.random() < 0.2)
        dropout_rate = self.dropout_rate
        if random.random() < constants.MUTATION_RATE:
            size = max(int(random.gauss(size, 2)), 1)
            activation = random.choice(constants.ACTIVATIONS)
            dropout_rate = max(min(random.gauss(dropout_rate, 0.1), 1), 0)
        return DenseUnit(size, activation, has_dropout, dropout_rate)

    def add_to_network(self, network: Sequential) -> None:
        network.add(Dense(self.size, activation=self.activation))
        if self.has_dropout:
            network.add(Dropout(self.dropout_rate))

    def __str__(self) -> str:
        string = "Dense size:{}\tactivation:{}".format(self.size, self.activation)
        if self.has_dropout:
            string += "\tdropout:{}".format(self.dropout_rate)
        return string

    def __eq__(self, o: 'DenseUnit') -> bool:
        if type(self) != type(o):
            return False

        return self.size == o.size and self.activation == o.activation and self.dropout_rate == o.dropout_rate
