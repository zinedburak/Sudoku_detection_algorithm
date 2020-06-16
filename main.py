"""Burak Deniz S010031 Department of Computer Science"""

import operator
import time

import mnist
import numpy as np
import glob
import cv2
import os
import natsort


# define your own functions here, Including MNIST and the functions from your previous assignment:

def pca(matrix):
    mean = np.mean(matrix)
    center = matrix - mean
    cov_matrix = np.cov(center.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    pairs = list()

    """traverse on both eigenvalues and eigenvectors and create a key value array where key is the eigenvalue and 
    value is the corresponding eigenvector """
    for i in range(len(eigenvalues)):
        pairs.append((eigenvalues[i], eigenvectors[i]))
    """sort the eigenvalue,eigenvector array according to eigenvalues """

    sorted_values = sorted(pairs, key=lambda t: t[0], reverse=True)

    """in order to calculate the variance we take the sum of all the eigenvalues and traverse on sorted key value 
    pairs of the eigenvalue and eigenvectors to take the eigenvectors until we reach the 0.9 variance"""
    current_sum_of_eigenvalues = 0
    sum_of_all_eigenvalues = sum(eigenvalues)
    array_eigenvectors = []
    count = 0
    array_eigenvalues = []
    for i in range(len(sorted_values)):
        array_eigenvectors.append(sorted_values[i][1])
        array_eigenvalues.append(sorted_values[i][0])
        current_sum_of_eigenvalues += sorted_values[i][0]
        count += 1
        if current_sum_of_eigenvalues / sum_of_all_eigenvalues >= 0.95:
            array_eigenvectors = np.array(array_eigenvectors)
            print('variance', current_sum_of_eigenvalues / sum_of_all_eigenvalues)
            break
    return array_eigenvectors


def euclidean_distance(longRow, shortRow):
    results = np.zeros(len(longRow))
    shortRow = shortRow.real
    # print('short row len ',len(shortRow))
    results[0:len(shortRow)] = shortRow
    return np.linalg.norm(longRow - results)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(train_row, test_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    # print(output_values)
    prediction = max(set(output_values), key=output_values.count)
    return prediction


"""will take a train set with labels and a test set without labels it will search each row of test set in train set 
according to knn function .  it will take a breakPoint from user and will stop after evaluating data until breakpoint
This evaluation function is only good for the data where we already know the labels of the data
"""


def evaluate_MNIST(trainDataSetW_Labels, testDataset, neighborCount, breakPoint, labels):
    error_count = 0
    print("Evaluating.....")
    for i in range(len(testDataset)):
        prediction = predict_classification(trainDataSetW_Labels, testDataset[i], neighborCount)
        print('Expected %d, predicted %d.' % (labels[i].real, prediction.real))
        if labels[i] != prediction:
            error_count += 1
        if i == breakPoint:
            print('Accuracy of MNIST is = ', 1 - error_count / breakPoint)
            break


"""Previous code from Assigment 1 - (Changed the code because finding contours was not enough for this assignment """

"""Basic code for preventing the repetation of cv2.imshow, cv2.waitkey """


def show_image(img):
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows


"""Preprocess the image for digit sudoku box detection"""


def preprocess_img(img, skip_dilate=False):
    preprocess = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    preprocess = cv2.bitwise_not(preprocess, preprocess)

    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        preprocess = cv2.dilate(preprocess, kernel)

    return preprocess


"""Get the corners of the sudoku puzzle"""


def get_corners_of_largest_poly(img):
    # Top Left: Smallest x and smallest y co-ordinate [minimise](x+y)
    # Top Right: Largest x and smallest y co-ordinate [maximise](x-y)
    # Bottom Left: Largest x and largest y co-ordinate [maximise](x+y)
    # Bottom Right: Smallest x and largest y co-ordinate [minimise](x-y)

    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending

    polygon = contours[0]  # get largest contour

    # operator.itemgetter - get index of point
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


"""This code extracts the sudoku puzzle and wraps it to a square like shape 
This process makes it easier to detect squares of the sudoku puzzle"""


def infer_sudoku_puzzle(image, crop_rectangle):
    img = image
    crop_rect = crop_rectangle

    def distance_between(a, b):
        return np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))

    def crop_img():
        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

        source_rect = np.array(np.array([top_left, bottom_left, bottom_right, top_right],
                                        dtype='float32'))  # float for perspective transformation

        side = max([
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right)
        ])

        dest_square = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        m = cv2.getPerspectiveTransform(source_rect, dest_square)

        return cv2.warpPerspective(img, m, (int(side), int(side)))

    return crop_img()


"""Since we know that the all sudoku pictures has 81 little boxes we represent each box with two points consist of 
top lef and bottom right corners which we find by dividing the images side to 9 to get approximate length of one box 
Basic image math nothing fancy"""


def infer_grid(img):  # infer 81 cells from image
    squares = []
    side = img.shape[:1]

    side = side[0] / 9

    for i in range(9):  # get each box and append it to squares -- 9 rows, 9 cols
        for j in range(9):
            p1 = (i * side, j * side)  # top left corner of box
            p2 = ((i + 1) * side, (j + 1) * side)  # bottom right corner of box
            squares.append((p1, p2))
    return squares


"""
All the digit recognition and extraction is done here
we take each rect that we found in infer grid function      
"""


def extract_data(in_img, rects, common_pca, train_images):
    img = in_img.copy()
    # get rects from top to bottom left to right

    result = []
    for rect in rects:  # get each rect that comes from infer grid
        x, y, w, h = int(rect[0][0]), int(rect[0][1]), int(rect[1][0] - rect[0][0]), int(rect[1][1] - rect[0][1])

        # crop the image so that we have only one sudoku box
        one_box = img[y:(y + h), x:(x + w)]

        # put the image to preprocess so that we will have a black and white pic where number and grid lines are
        # white and background is black
        one_box = preprocess_img(one_box)

        # resize the image to 28 X 28 so we can apply the common pca to the picture which is 154 X 784 in 0.95 variance
        resized_oneBox = cv2.resize(one_box, (28, 28), interpolation=cv2.INTER_AREA)

        # we will take the mean of each box if it is near black we would assume that it has no digit in it
        mean_oneBox = np.mean(resized_oneBox)

        # we will resize the image so that we can apply common pca to it and project train data set and sudoku box in
        # to the same space
        rowData_oneBox = np.resize(resized_oneBox, (1, 784))

        # apply common pca to row data to create 1 X 154 in 0.95 variance
        pcaData_onebox = rowData_oneBox.dot(common_pca.T)
        # if the mean of the pixels in sudoku box is less then 60 it has no number other wise detect the number and
        # append to the result set
        if mean_oneBox <= 60:
            result.append(0)
        elif mean_oneBox > 60:
            result.append(predict_classification(train_images, pcaData_onebox[0], 5).real)
            # show_image(one_box )
    # since the order of the squares array from  infer grid is top the bottom first left to right after we take the
    # result set reshape it into 9 X 9 array and transpose it so we have left to right bottom the lef data
    result = np.array(result)
    result = result.reshape(9, 9)
    result = result.T
    return result


def SudokuDigitDetector(img, pca, trainDataset):
    processed_sudoku = preprocess_img(img)
    corners_of_sudoku = get_corners_of_largest_poly(processed_sudoku)
    cropped_sudoku = infer_sudoku_puzzle(img, corners_of_sudoku)
    squares_on_sudoku = infer_grid(cropped_sudoku)
    return extract_data(cropped_sudoku, squares_on_sudoku, pca, trainDataset)



def sudokuAcc(gt, out):
    return (gt == out).sum() / gt.size * 100


if __name__ == "__main__":
    # change the path accourding to
    path = 'C:/Users/Lenovo/PycharmProjects/Assigment2-NotComplete/samples'
    # MNIST experiments:

    """Data Load"""
    mndata = mnist.MNIST(path)
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    "casting train images and labels as np array"
    trainImages_numpy_array = np.array(train_images)
    train_labels = np.array(train_labels)

    "casting test images and labels as a np array "
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    """Code for mnist evaluate"""

    common_pca = pca(trainImages_numpy_array)

    train_images = trainImages_numpy_array.dot(common_pca.T)
    train_images = train_images.T
    train_images = np.vstack([train_images, train_labels])
    train_images = train_images.T
    test_images = test_images.dot(common_pca.T)

    """will evaluate the first 10 digits of the test images change it in order to get larger evaluations"""
    evaluate_MNIST(train_images, test_images, 7, 10, test_labels)

    # Sudoku Experiments:

    experiment_imagePath = "images/image1.jpg"
    experiment_Image = cv2.imread(experiment_imagePath, cv2.IMREAD_GRAYSCALE)
    experiment_dataPath = 'images/image1.dat'
    experiment_gt = np.genfromtxt(experiment_dataPath, skip_header=2, dtype=int, delimiter=' ')
    print("Extracting the data from sudoku....")
    print('real data ',experiment_gt)
    experiment_output = SudokuDigitDetector(experiment_Image, common_pca, train_images)
    print('experiment output',experiment_output)
    print('Accuracy of image 1 is ', sudokuAcc(experiment_gt, experiment_output))
    """"""
    image_dirs = 'images/*.jpg'
    data_dirs = 'images/*.dat'
    IMAGE_DIRS = natsort.natsorted(glob.glob(image_dirs))
    DATA_DIRS = natsort.natsorted(glob.glob(data_dirs))
    total_acc = 0
    # Loop over all images and ground truth

    s = time.time()
    "when you run the for loop for all the images it will take around 100 minutes to complete and will produce around " \
    "51.1% accuracy"
    for i, (img_dir, data_dir) in enumerate(zip(IMAGE_DIRS, DATA_DIRS)):
        # print(img_dir, data_dir)
        # Define your variables etc.:
        image_name = os.path.basename(img_dir)
        gt = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        # print(gt)
        output = SudokuDigitDetector(img, common_pca, train_images)
        total_acc = total_acc + sudokuAcc(gt, output)
        print("Sudoku dataset accuracy: {}".format(total_acc / (i + 1)))
    print(time.time() - s)
