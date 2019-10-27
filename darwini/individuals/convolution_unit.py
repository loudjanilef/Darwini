import random

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

import darwini.constants as constants
from darwini.individuals.individual_unit import IndividualUnit


class ConvolutionUnit(IndividualUnit):
    filters_nbr: int
    kernel_size: int
    stride: int
    activation: str

    has_pooling: bool
    pooling_size: int
    pooling_stride: int

    def __init__(self, filters_nbr: int, kernel_size: int, stride: int, activation: str, has_pooling: bool,
                 pooling_size: int, pooling_stride: int) -> None:
        self.filters_nbr = filters_nbr
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.has_pooling = has_pooling
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride

    @staticmethod
    def generate():
        filters_nbr = random.randint(constants.MIN_CONV_FILTERS, constants.MAX_CONV_FILTERS)
        kernel_size = random.randint(constants.MIN_CONV_KERNEL_SIZE, constants.MAX_CONV_KERNEL_SIZE)
        stride = random.randint(constants.MIN_CONV_STRIDE, constants.MAX_CONV_STRIDE)
        activation = random.choice(constants.ACTIVATIONS)
        has_pooling = random.random() < constants.POOLING_PROBABILITY
        pooling_size = random.randint(constants.MIN_POOL_SIZE, constants.MAX_POOL_SIZE)
        pooling_stride = max(pooling_size - abs(random.gauss(pooling_size, 2)), 1)
        return ConvolutionUnit(filters_nbr, kernel_size, stride, activation, has_pooling, pooling_size, pooling_stride)

    def blend(self, partner: 'ConvolutionUnit') -> 'ConvolutionUnit':
        filters_nbr = random.choice([self.filters_nbr, partner.filters_nbr])
        kernel_size = random.choice([self.kernel_size, partner.kernel_size])
        stride = random.choice([self.stride, partner.stride])
        activation = random.choice([self.activation, partner.activation])
        has_pooling = random.choice([self.has_pooling, partner.has_pooling])
        pooling_size = random.choice([self.pooling_size, partner.pooling_size])
        pooling_stride = random.choice([self.pooling_stride, partner.pooling_size])
        return ConvolutionUnit(filters_nbr, kernel_size, stride, activation, has_pooling, pooling_size, pooling_stride)

    def mutate(self) -> 'ConvolutionUnit':
        filters_nbr = self.filters_nbr
        kernel_size = self.kernel_size
        stride = self.stride
        activation = self.activation
        has_pooling = self.has_pooling
        pooling_size = self.pooling_size
        pooling_stride = self.pooling_stride

        if random.random() < constants.MUTATION_RATE:
            filters_nbr = max(int(random.gauss(filters_nbr, 2)), 1)
            kernel_size = max(int(random.gauss(kernel_size, 2)), 1)
            stride = max(int(random.gauss(stride, 1)), 0)
            activation = random.choice(constants.ACTIVATIONS)
            has_pooling = random.random() < constants.POOLING_PROBABILITY
            pooling_size = max(int(random.gauss(pooling_size, 1)), 1)
            pooling_stride = max(int(random.gauss(pooling_stride, 1)), 1)
        return ConvolutionUnit(filters_nbr, kernel_size, stride, activation, has_pooling, pooling_size, pooling_stride)

    def add_to_network(self, network: Sequential, data_format='channels_last', input_shape=None) -> None:
        if input_shape is not None:
            network.add(
                Conv2D(self.filters_nbr, (self.kernel_size, self.kernel_size), strides=(self.stride, self.stride),
                       data_format=data_format, padding='same', activation=self.activation, input_shape=input_shape))
        else:
            network.add(
                Conv2D(self.filters_nbr, (self.kernel_size, self.kernel_size), strides=(self.stride, self.stride),
                       data_format=data_format, padding='same', activation=self.activation))
        if self.has_pooling:
            network.add(
                MaxPooling2D((self.pooling_size, self.pooling_stride), padding='same',
                             strides=(self.pooling_stride, self.pooling_stride)))

    def __eq__(self, o: 'ConvolutionUnit') -> bool:
        if type(self) != type(o):
            return False
        equals = self.filters_nbr == o.filters_nbr and self.kernel_size == o.kernel_size and self.stride == o.stride \
                 and self.activation == o.activation and self.has_pooling == o.has_pooling \
                 and self.pooling_size == o.pooling_size and self.pooling_stride == o.pooling_stride
        return equals

    def __str__(self) -> str:
        string = "Conv filters:{}\tsize:{}\tstride:{}\tactivation:{}".format(self.filters_nbr, self.kernel_size,
                                                                             self.stride, self.activation)
        if self.has_pooling:
            string += "\nPooling size:{}\tstride:{}".format(self.pooling_size, self.pooling_stride)
        return string
