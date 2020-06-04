import numpy as np
import skimage

class Pooling:

    def __init__(self):
        self.window_size = 2
        self.stride = 2

    def walk(self, image):
        r, c = image.shape
        nr = r // 2
        nc = c // 2
        output = np.zeros((nr, nc))
        for i in range(nr):
            for j in range(nc):
                cur_r = i * self.stride
                cur_c = j * self.stride
                window = image[cur_r : (cur_r + self.window_size), cur_c : (cur_c + self.window_size)]
                output[i, j] = np.amax(window)
        return output
    
    def forward(self, filtered_images):
        n, x, x = filtered_images.shape
        output = list()
        for x in range(n):
            image = filtered_images[x]
            output.append(self.walk(image))
            
        return np.array(output)

# img = np.array([
#     [0.77, 0, 0.11, 0.33, 0.55, 0],
#     [0, 1, 0, 0.33, 0, 0.11],
#     [0.11, 0, 1, 0, 0.11, 0],
#     [0.33, 0.33, 0, 0.55, 0, 0.33]])
# convLayer = Pooling()
# print(convLayer.forward(img))