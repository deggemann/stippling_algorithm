import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import scipy.signal as sig   
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import LinearNDInterpolator, griddata

from PIL import Image
import numpy as np
import random as rnd

def get_coordinates(n_points, shift_x, shift_y, region_size):
    x = [(rnd.random() + shift_x)*region_size for _ in range(n_points)]
    y = [(rnd.random() + shift_y)*region_size for _ in range(n_points)]

    return list(zip(x,y))

def stipple_image(sampling_size: int, n_groups: int):
    kernel_average = np.ones((sampling_size,sampling_size))/(sampling_size**2)
    convolution_tmp = sig.convolve(img, kernel_average)

    # only keep every sampling_size "n" value
    convolution = convolution_tmp[::sampling_size, ::sampling_size]

    print(f"Size of original image: {img.shape}")
    print(f"SIze of convolution: {convolution.shape}")

    # use value of pixel average to generate n coordinates for each region of the image
    conv_height, conv_width = convolution.shape
    coordinates = []
    for column in range(conv_width):
        for row in range(conv_height):
            n_points = int((255-convolution[row, column]) * n_groups / 255)
            coordinates.extend(get_coordinates(n_points, column, conv_height-row, sampling_size))
    
    coordinates = np.array(coordinates)
    print(f"Number of dots: {coordinates.shape}")
    return coordinates

def weight_function(img, upsampling):
    x_size, y_size = img.shape

    x = np.array(list(np.arange(0,y_size)) * x_size)
    y = np.array(list(np.repeat(np.arange(0,x_size), y_size)))

    X = np.arange(0, y_size, upsampling)
    Y = np.arange(0, x_size, upsampling)
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

    Z = griddata((x, y), img.reshape(-1), (X, Y), method='cubic')
    return Z    

def calculate_point_shift(voronoi, weight):
    new_points = []

    max_y, max_x = weight.shape

    for region_index in voronoi.point_region:
        vertices_indexes = voronoi.regions[region_index]
        vertices = [voronoi.vertices[index] for index in vertices_indexes]
        
        mass_distance = [0, 0]
        tot_mass = 0
        for point in vertices:
            point_int = [round(coord) for coord in point]
            point_int[0] = sorted([0, point_int[0], max_x-1])[1]
            point_int[1] = sorted([0, point_int[1], max_y-1])[1]

            mass_distance[0] += point_int[0] * float(weight[max_y-1 - point_int[1], point_int[0]])
            mass_distance[1] += point_int[1] * float(weight[max_y-1 - point_int[1], point_int[0]])
            tot_mass += int(weight[max_y-1 - point_int[1], point_int[0]])

        com = [coord/tot_mass for coord in mass_distance]
        new_points.append(com)

    return np.array(new_points)

if __name__ == "__main__":
    PATH = "portrait.jpg"
    SAMPLING_SIZE = 10
    GROUPS_NMBR = 10

    # img = mpimg.imread(PATH)

    img = np.array(Image.open(PATH).convert('L'))
    WIDTH, HEIGHT = img.shape

    coordinates = stipple_image(SAMPLING_SIZE, GROUPS_NMBR)

    stippling_img = plt.scatter(coordinates[:,0], coordinates[:,1],s=3)
    plt.show()

    plt.imshow(img, cmap="bone")
    plt.show()

    # compute weight distribution
    # upsampling = 1
    # weight = weight_function(img, upsampling)
    # plt.imshow(weight)
    # plt.show()

    for i in range(10):
        print(f"Iteration number {i}")
        # get voronoi diagram
        vor = Voronoi(coordinates)
        
        # get point shifts
        coordinates = calculate_point_shift(vor, img)

    stippling_img = plt.scatter(coordinates[:,0], coordinates[:,1],s=3)
    plt.show()