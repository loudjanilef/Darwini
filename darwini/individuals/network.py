import random
from typing import List

from fastdtw import fastdtw
from keras.layers import Flatten, Dense
from keras.models import Sequential

from darwini.individuals.convolution_unit import ConvolutionUnit
from darwini.individuals.dense_unit import DenseUnit
from darwini.individuals.individual import Individual
from darwini.individuals.individual_unit import IndividualUnit


def adjust_size(units: List, desired_len: int, new_unit: IndividualUnit) -> List:
    while desired_len < len(units):
        if len(units) == 0:
            break
        remove_index = random.choice(range(len(units)))
        units.pop(remove_index)
    while desired_len > len(units):
        add_index = random.choice(range(len(units))) if len(units) > 0 else 0
        units.insert(add_index, new_unit)
    return units


def blend_lists(self_units: List, partner_units: List) -> List:
    self_len, partner_len = len(self_units), len(partner_units)
    units_nbr = random.choice([self_len, partner_len])
    if units_nbr == 0:
        return []
    selected_indexes = random.sample(range(max(self_len, partner_len)), units_nbr)
    units = []
    for index in selected_indexes:
        if index < self_len and index < partner_len:
            units.append(self_units[index].blend(partner_units[index]))
        elif index < self_len:
            units.append(self_units[index])
        else:
            units.append(partner_units[index])
    return units


def blend_convs(self_units: List[ConvolutionUnit], partner_units: List[ConvolutionUnit]) -> List:
    def distance(u: int, v: int):
        return abs(u - v)

    self_blocks = []
    partner_blocks = []
    self_output_sizes = [unit.output_size() for unit in self_units]
    partner_output_sizes = [unit.output_size() for unit in partner_units]
    _, path = fastdtw(self_output_sizes, partner_output_sizes, dist=distance)
    last_self_index, last_partner_index = -1, -1
    self_block, partner_block = [], []
    first_elem = True
    for self_index, partner_index in path:
        if not first_elem and self_index != last_self_index and partner_index != last_partner_index:
            partner_blocks.append(partner_block)
            partner_block = []
            self_blocks.append(self_block)
            self_block = []
        if self_index != last_self_index:
            self_block.append(self_units[self_index])
        if partner_index != last_partner_index:
            partner_block.append(partner_units[partner_index])
        last_self_index = self_index
        last_partner_index = partner_index
        first_elem = False
    partner_blocks.append(partner_block)
    self_blocks.append(self_block)

    # Select random matching blocks from the two lists
    blocks = []
    for self_block, partner_block in zip(self_blocks, partner_blocks):
        blocks.append(random.choice([self_block, partner_block]))
    # Flatten blocks
    units = [item for sublist in blocks for item in sublist]
    # Update input sizes
    input_size = units[0].output_size()
    for i in range(1, len(units)):
        units[i].input_size = input_size
        input_size = units[i].output_size()
    return units


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
        dense_units_nbr = random.randint(0, 10)
        conv_units = []
        input_size = input_shape[0]
        while True:
            conv_unit = ConvolutionUnit.generate(input_size)
            conv_units.append(conv_unit)
            output_size = conv_unit.output_size()
            if output_size <= 8:
                break
            input_size = output_size
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
        conv_units = blend_convs(self.conv_units, partner.conv_units)
        dense_units = blend_lists(self.dense_units, partner.dense_units)
        return Network(self.input_shape, self.output_shape, self.data_format, conv_units, dense_units)

    def mutate(self) -> 'Network':
        desired_new_dense_len = max(int(random.gauss(len(self.dense_units), 1)), 0)
        conv_units = [conv.mutate() for conv in self.conv_units]
        dense_units = [dense.mutate() for dense in self.dense_units]
        dense_units = adjust_size(dense_units, desired_new_dense_len, DenseUnit.generate())
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
        model.add(Dense(self.output_shape, activation="relu"))
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        return model

    def save(self, filename) -> None:
        with open(filename, "w+") as file:
            file.write(str(self))

    def __eq__(self, o: 'Network') -> bool:
        if type(self) != type(o):
            return False

        if self.data_format != o.data_format or self.input_shape != o.input_shape or \
                self.output_shape != o.output_shape or len(self.conv_units) != len(o.conv_units) \
                or len(self.dense_units) != len(o.dense_units):
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
