import random
from typing import List

from keras.layers import Flatten, Dense
from keras.models import Sequential

from darwini.individuals.convolution_unit import ConvolutionUnit
from darwini.individuals.dense_unit import DenseUnit
from darwini.individuals.individual import Individual


class Network(Individual):
    input_shape: List[int]
    output_shape: int
    data_format: str

    conv_units: List[ConvolutionUnit]
    dense_units: List[DenseUnit]

    @staticmethod
    def generate(input_shape: List[int] = None, output_shape: int = None,
                 data_format='channels_last') -> 'Network':
        if input_shape is None or output_shape is None:
            raise ValueError("Generation of a network needs input and output shapes")
        conv_units_nbr = random.randint(0, 10)
        dense_units_nbr = random.randint(0, 10)
        conv_units = []
        for _ in range(conv_units_nbr):
            conv_units.append(ConvolutionUnit.generate())
        dense_units = []
        for _ in range(dense_units_nbr):
            dense_units.append(DenseUnit.generate())
        return Network(input_shape, output_shape, data_format, conv_units, dense_units)

    def __init__(self, input_shape: List[int], output_shape: int, data_format: str,
                 conv_units: List[ConvolutionUnit], dense_units: List[DenseUnit]) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_format = data_format
        self.conv_units = conv_units
        self.dense_units = dense_units

    def blend(self, partner: 'Network') -> 'Network':
        self_conv_len, self_dense_len = len(self.conv_units), len(self.dense_units)
        partner_conv_len, partner_dense_len = len(partner.conv_units), len(partner.dense_units)
        conv_units_nbr = random.choice([self_conv_len, partner_conv_len])
        dense_units_nbr = random.choice([self_dense_len, partner_dense_len])
        selected_conv_units = random.sample(range(max(self_conv_len, partner_conv_len)), conv_units_nbr)
        selected_dense_units = random.sample(range(max(self_dense_len, partner_dense_len)), dense_units_nbr)
        conv_units = []
        for index in selected_conv_units:
            if index < self_conv_len and index < partner_conv_len:
                conv_units.append(self.conv_units[index].blend(partner.conv_units[index]))
            elif index < self_conv_len:
                conv_units.append(self.conv_units[index])
            else:
                conv_units.append(partner.conv_units[index])
        dense_units = []
        for index in selected_dense_units:
            if index < self_dense_len and index < partner_dense_len:
                dense_units.append(self.dense_units[index].blend(partner.dense_units[index]))
            elif index < self_dense_len:
                dense_units.append(self.dense_units[index])
            else:
                dense_units.append(partner.dense_units[index])
        return Network(self.input_shape, self.output_shape, self.data_format, conv_units, dense_units)

    def mutate(self) -> 'Network':
        desired_new_conv_len = max(int(random.gauss(len(self.conv_units), 1)), 0)
        desired_new_dense_len = max(int(random.gauss(len(self.dense_units), 1)), 0)
        conv_units = [conv.mutate() for conv in self.conv_units]
        dense_units = [dense.mutate() for dense in self.dense_units]
        while desired_new_conv_len < len(conv_units):
            remove_index = random.choice(range(len(conv_units)))
            conv_units.pop(remove_index)
        while desired_new_conv_len > len(conv_units):
            add_index = random.choice(range(len(conv_units)))
            conv_units.insert(add_index, ConvolutionUnit.generate())
        while desired_new_dense_len < len(dense_units):
            remove_index = random.choice(range(len(dense_units)))
            dense_units.pop(remove_index)
        while desired_new_dense_len > len(dense_units):
            add_index = random.choice(range(len(dense_units)))
            dense_units.insert(add_index, DenseUnit.generate())
        return Network(self.input_shape, self.output_shape, self.data_format, conv_units, dense_units)

    def compile(self) -> Sequential:
        model = Sequential()
        first = True
        for unit in self.conv_units:
            if first:
                unit.add_to_network(model, self.data_format, self.input_shape)
                first = False
                continue
            unit.add_to_network(model, self.data_format)
        model.add(Flatten())
        for unit in self.dense_units:
            unit.add_to_network(model)
        model.add(Dense(self.output_shape))
        model.compile('adagrad', 'categorical_crossentropy', metrics=['accuracy'])
        return model

    def save(self, filename) -> None:
        with open(filename, "w+") as file:
            file.write(str(self))

    def __eq__(self, o: 'Network') -> bool:
        if type(self) != type(o):
            return False

        if self.data_format != o.data_format or self.input_shape != o.input_shape or self.output_shape != o.output_shape \
                or len(self.conv_units) != len(o.conv_units) or len(self.dense_units) != len(o.dense_units):
            return False

        for self_conv, o_conv in zip(self.conv_units, o.conv_units):
            if self_conv != o_conv:
                return False

        for self_dense, o_dense in zip(self.dense_units, o.dense_units):
            if self_dense != o_dense:
                return False
        return True

    def __str__(self) -> str:
        string = "Network input:{}".format(self.input_shape)
        for conv in self.conv_units:
            string += "\n" + str(conv)
        string += "\nFlatten"
        for dense in self.dense_units:
            string += "\n" + str(dense)
        string += "\noutput:{}".format(self.output_shape)
        return string
