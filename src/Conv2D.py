import numpy as np

class Conv2D:

    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def do_filter(self, image, filter):
        num_rows, num_cols = image.shape
        output = np.zeros((num_rows - 2, num_cols - 2))

        for i in range(num_rows - 2):
            for j in range(num_cols - 2):
                im = image[i:(i + 3), j:(j + 3)] * filter
                output[i,j] = np.sum(im)
            
        return output

    def forward(self, image):
        output = list()
        for filter in self.filters:
            output.append(self.do_filter(image, filter))
        return np.array(output)