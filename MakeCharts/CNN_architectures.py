import pandas as pd
import matplotlib.pyplot as plt

class Dimensions:
    def __init__(self, dimensions_str: str, separator):
        '''
        dimensions_str:     String with all the dimensions. Ex: 227_3_11_96
        separator:          Character that separate each dimension
        '''
        self.MAX_DIFFERENCE = 5
        self.dimensions = [int(d) for d in dimensions_str.split('_')]
        self.separator = separator

    def __str__(self):
        return self.separator.join([str(d) for d in self.dimensions])

    def __hash__(self):
        return hash((self.dimensions[0], self.dimensions[1], self.dimensions[2], self.dimensions[3]))

    def __eq__(self, other):
        '''
        operator ==. Two dimensions are equal if the difference of all the internal
        values is less than the self.max_difference value.
        '''
        if len(self.dimensions) != len(other.dimensions): return False
        for i in range(len(self.dimensions)):
            if abs((self.dimensions[i]) - other.dimensions[i]) > self.MAX_DIFFERENCE: return False
        return True
    
    def __neq__(self, other):
        return self != other



class CNN:
    def __init__(self, name: str, dimensions: list):
        '''
        name:       Name of the architecture
        dimensions: List with the dimensions of the architecture
        '''
        self.name = name
        self.dimensions = []
        for d in dimensions:
            self.dimensions.append(Dimensions(d, separator='_'))

class CNN_visualizer:
    def __init__(self, CNNs: list):
        '''
        CNNs:   List of CNNs
        '''
        self.dimensions = {}
        self.dimensions_reduced = {}
        self.CNNs = {}
        for CNN in CNNs:
            self.CNNs[CNN.name] = CNN
            for d in CNN.dimensions:
                if d not in self.dimensions:
                    self.dimensions[d] = 1
                else:
                    self.dimensions[d] += 1

        # for d in self.dimensions:
        #     print(d)
        dimensions_list = [d for d in self.dimensions.keys()]

        plt.bar([str(d) for d in self.dimensions.keys()], self.dimensions.values())


        plt.show()


if __name__ == '__main__':
    AlexNet = CNN('AlexNet', [
        '227_3_11_96', '27_96_5_256', '13_256_3_384', '13_384_3_384', '13_384_3_256'])
    ResNet50 = CNN('ResNet50', [
        '229_3_7_64', '112_64_1_64', '112_64_3_64', '56_64_1_256',
        '56_256_1_128', '56_128_3_128', '28_128_1_512',
        '28_512_1_256', '28_256_3_256', '14_256_1_1024',
        '14_1024_1_512', '14_512_3_512', '7_512_1_2048'
    ])
    my_CNN_visualizer = CNN_visualizer([AlexNet, ResNet50])
    # print(str(my_CNN.dimensions[1]))
    # my_dimensions_1 = Dimensions("227_3_11_96", separator='_',)
    # my_dimensions_2 = Dimensions("227_3_11_90", separator='_')
    # print(my_dimensions_1 == my_dimensions_2)
