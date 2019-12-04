import math
import random

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

import darwini.constants as constants
from darwini.individuals.individual_unit import IndividualUnit


class ConvolutionUnit(IndividualUnit):
    filters_nbr: int
    kernel_size: int
    activation: str

    has_pooling: bool
    pooling_size: int

    input_size: int

    def __init__(self, input_size: int, filters_nbr: int, kernel_size: int, activation: str, has_pooling: bool,
                 pooling_size: int) -> None:
        self.filters_nbr = filters_nbr
        self.kernel_size = kernel_size
        self.activation = activation
        self.has_pooling = has_pooling
        self.pooling_size = pooling_size
        self.input_size = input_size

    @staticmethod
    def generate(input_size: int = 0):
        filters_nbr = random.randint(constants.MIN_CONV_FILTERS, constants.MAX_CONV_FILTERS)
        kernel_size = random.randint(constants.MIN_CONV_KERNEL_SIZE, constants.MAX_CONV_KERNEL_SIZE)
        activation = random.choice(constants.ACTIVATIONS)
        has_pooling = random.random() < constants.POOLING_PROBABILITY
        pooling_size = random.randint(constants.MIN_POOL_SIZE, constants.MAX_POOL_SIZE)
        return ConvolutionUnit(input_size, filters_nbr, kernel_size, activation, has_pooling, pooling_size)

    def blend(self, partner: 'ConvolutionUnit') -> 'ConvolutionUnit':
        filters_nbr = random.choice([self.filters_nbr, partner.filters_nbr])
        kernel_size = random.choice([self.kernel_size, partner.kernel_size])
        activation = random.choice([self.activation, partner.activation])
        has_pooling = random.choice([self.has_pooling, partner.has_pooling])
        pooling_size = random.choice([self.pooling_size, partner.pooling_size])
        return ConvolutionUnit(self.input_size, filters_nbr, kernel_size, activation, has_pooling, pooling_size)

    def mutate(self) -> 'ConvolutionUnit':
        filters_nbr = self.filters_nbr
        kernel_size = self.kernel_size
        activation = self.activation
        has_pooling = self.has_pooling
        pooling_size = self.pooling_size

        if random.random() < constants.MUTATION_RATE:
            filters_nbr = max(int(random.gauss(filters_nbr, 2)), 1)
        return ConvolutionUnit(self.input_size, filters_nbr, kernel_size, activation, has_pooling, pooling_size)

    def add_to_network(self, network: Sequential, data_format='channels_last', input_shape=None) -> None:
        if input_shape is not None:
            network.add(
                Conv2D(self.filters_nbr, (self.kernel_size, self.kernel_size), strides=(1, 1),
                       data_format=data_format, activation=self.activation, input_shape=input_shape))
        else:
            network.add(
                Conv2D(self.filters_nbr, (self.kernel_size, self.kernel_size), strides=(1, 1),
                       data_format=data_format, activation=self.activation))
        if self.has_pooling:
            network.add(
                MaxPooling2D((self.pooling_size, self.pooling_size), strides=(self.pooling_size, self.pooling_size)))

    def output_size(self):
        output_size = math.ceil(self.input_size - self.kernel_size + 1)
        if self.has_pooling:
            output_size = math.ceil(output_size / self.pooling_size)
        return output_size

    def __eq__(self, o: 'ConvolutionUnit') -> bool:
        if type(self) != type(o):
            return False
        equals = self.filters_nbr == o.filters_nbr and self.kernel_size == o.kernel_size \
                 and self.activation == o.activation and self.has_pooling == o.has_pooling \
                 and self.pooling_size == o.pooling_size
        return equals

    def __str__(self) -> str:
        string = "Conv filters:{}\tsize:{}\tactivation:{}".format(self.filters_nbr, self.kernel_size, self.activation)
        if self.has_pooling:
            string += "\nPooling size:{}".format(self.pooling_size)
        string += "\tOutput size:{}".format(self.output_size())
        return string
