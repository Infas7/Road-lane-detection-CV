import cv2
import numpy as np
import glob
import re
import os

def generate_video_output(output_folder, output_video_file):
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    images = []
    files = []
    for file in glob.glob(output_folder + '/*.jpg'):
        files.append(file)
        img = cv2.imread(file)

    files.sort(key=natural_keys)
          
    for file in files:  
        img = cv2.imread(file)
        height, width, layers = img.shape
        size = (width,height)
        images.append(img)

    video_out = cv2.VideoWriter( output_video_file + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    
    for i in range(len(images)):
        video_out.write(images[i])
    video_out.release()


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def slope_average(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if not left_fit or not right_fit:
        return np.array([])

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return np.array([left_line, right_line])


def sobel_edge_detection(image, edge_thresh_ratio=0.4):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    
    threshold = np.max(magnitude) * edge_thresh_ratio
    edge = np.uint8((magnitude > threshold) * 255)
    return edge



def get_line_intersection(lines):
    x_diff = (lines[0][0] - lines[0][2], lines[1][0] - lines[1][2])
    y_diff = (lines[0][1] - lines[0][3], lines[1][1] - lines[1][3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
       return 0, 0

    d = (det(*((lines[0][0],lines[0][1]),(lines[0][2],lines[0][3]))), det(*((lines[1][0],lines[1][1]),(lines[1][2],lines[1][3]))))
    x = int(det(d, x_diff) / div)
    y = int(det(d, y_diff) / div)

    return x, y


def generate_lines(image, lines):
    end_x,end_y = get_line_intersection(lines)

    lanes = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lanes, (x1, y1), (end_x, end_y), (255, 0, 0), 5)

    cv2.circle(lanes, (end_x,end_y), 10, (255, 255, 0), 2)
    return lanes

def get_process_area(image):
    polygons = np.array([[(0, image.shape[0]),(0, image.shape[0]-100), (290, 320), (340, 320), (image.shape[1]-150, image.shape[0])]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(edge_image, theta_res=1, rho_res=1):
    height, width = edge_image.shape
    max_rho = int(np.hypot(height, width))
    rhos = np.arange(-max_rho, max_rho, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * max_rho, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edge_image)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + max_rho)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def find_peaks_in_accumulator(accumulator, threshold):
    peaks = []
    for rho, theta_values in enumerate(accumulator):
        for theta, value in enumerate(theta_values):
            if value > threshold:
                peaks.append((rho, theta))
    return peaks


def peaks_to_lines(peaks, thetas, rhos):
    lines = []
    for rho_idx, theta_idx in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines.append([x1, y1, x2, y2])
    return lines


def detect_lanes(img):
    edge_image = sobel_edge_detection(img)
    cropped_image = get_process_area(edge_image)
    
    accumulator, thetas, rhos = hough_transform(cropped_image)
    peaks = find_peaks_in_accumulator(accumulator, threshold=100)
    lines = peaks_to_lines(peaks, thetas, rhos)

    filtered_lines = slope_average(img, lines)
    lanes = generate_lines(img, filtered_lines)

    final_img = cv2.addWeighted(img, 0.8, lanes, 1, 1)
    return final_img



if __name__=="__main__":
    input_folder = 'TestVideo_1'
    output_folder = 'output-right'
    output_video_file = 'output-right'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    paths = glob.glob( input_folder + '/*.bmp')

    for i,image_path in enumerate(paths):
        print(image_path)
        image = cv2.imread(image_path)
        img = np.copy(image)
        result = detect_lanes(img)
        cv2.imwrite(output_folder + '/'+image_path[12:-4]+'_out.jpg', result)

    generate_video_output(output_folder, output_video_file)