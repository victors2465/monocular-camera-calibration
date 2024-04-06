'''

File name: get-measuraments.py 
Description: This script performs a measuerment of a selected points, also calibrates the camera

Author: Victor Santiago Solis Garcia
Creation date: 04/05/2024
Last update: 

Usage example:  python get-measuraments.py --cal_file calibration_data.json --cam_index 0 --Z 143

                python3 get-measuraments.py --cal_file calibration_data.json --cam_index 0 --Z 143

'''


'''
Import standard libraries 
'''
import cv2
import json
import numpy as np
import math 
import argparse
import sys 
import keyboard
import time 
from numpy.typing import NDArray


'''
Define global variables
'''
points:NDArray = [] 
frame:NDArray = None
points_real:NDArray=[]
distance_array:NDArray = []
data:NDArray = None
x:np.intc = None
y:np.intc = None
z:np.intc = None
middle_button_pressed = False
calibration = False

def load_calibration(calibration_data:str)->NDArray:
    '''
    Function to load the .json file to calibrate the camera 
    Parameters:    calibration_data(str): .json file

    Returns:       data(NDArray): Contains camera_matrix and dist_coeffs
    '''
    with open(calibration_data) as f:
        data = json.load(f)
    return data

def undistort_image(frame:NDArray, camera_matrix:NDArray, dist_coeffs:NDArray):
    '''
    Function to undistord the video frames
    Parameters:    calibration_data(str): .json file

    Returns:       data(NDArray): Contains camera_matrix and dist_coeffs
    '''
    global calibration
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                            dist_coeffs, 
                                                            (w, h), 
                                                            0, 
                                                            (w, h))
    undistorted_frame = cv2.undistort(frame, 
                                      camera_matrix, 
                                      dist_coeffs, 
                                      None, 
                                      new_camera_matrix)
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    if calibration == False:
        print(f'Camera calibrated')
        calibration = True

    return undistorted_frame

def parser_user_data()->argparse:
    '''
    Function to receive the user data 
    Parameters:    None

    Returns:       args(argparse): Argparse object with the cam index,
                                    the Z distance and the calibration file 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_index',
                        required=True,
                        type=int,
                        help='camera index (0,1,2)')
    parser.add_argument('--Z',
                        required=True,
                        type=float,
                        help='distance of the camera to the surface to meassure')
    parser.add_argument('--cal_file',
                        required=True,
                        type=str,
                        help='calibration file')
    
    args = parser.parse_args()
    return args

def calculate_real_coord(data:NDArray,X:np.intc,Y:np.intc,Z)->None:
    '''
    Function to calculate the real coordinate of the pixel using the calibration info. 
    Parameters:     data(NDArray): Calibration data
                    X(intc): X coordinate of the pixel
                    Y(intc): Y coordinate of the pixel
                    Z(intc): Real distance between the camero to the surface to meassure 

    Returns:       None
    '''
    ui = X 
    vi = Y
    Z = Z
    camera_matrix = np.array(data['camera_matrix'])
    fx = camera_matrix[0, 0]
    cx = camera_matrix[0, 2]
    fy = camera_matrix[1, 1]
    cy = camera_matrix[1, 2]
    X_real = (ui - cx)*(Z/fx)/100
    Y_real = (vi - cy)*(Z/fy)/100
    print(f"3D point coordinate X:{X_real}, Y:{Y_real}, Z:{Z}\n")
    points_real.append((X_real,Y_real))

def compute_line_segments(points_real:NDArray)->float:
    '''
    Function to load calculate the distance between the differents points and 
    order them in ascending order.
    Parameters:    points_real(NDArray): points with the real distance of the seletected pixel

    Returns:       perimeter(float): float objetct with the sum of all the distances
    '''
    perimeter = 0
    distance_array = [] 
    for i in range(1, len(points_real)):
        distance = math.sqrt((points_real[i-1][0] - points_real[i][0])**2 + 
                             (points_real[i-1][1] - points_real[i][1])**2)
        
        distance_array.append(distance)

        perimeter += distance

    last_to_first_distance = math.sqrt((points_real[-1][0] - points_real[0][0])**2 + 
                                       (points_real[-1][1] - points_real[0][1])**2)
    distance_array.append(last_to_first_distance)
    perimeter += last_to_first_distance
    sorted_distances = sorted(distance_array)

    # Assuming you want to print or otherwise process sorted distances separately
    if middle_button_pressed == False:
        for i, segment_length in enumerate(distance_array, 1):
            if i < len(points_real):
                print(f"Distance P{i-1},P{i}={segment_length}")
            elif i>2:
                print(f"Distance P{i-1},P0={segment_length}")
                print("\nSorted Distances:")
                for i, distance in enumerate(sorted_distances, 1):
                    print(f'{distance}')

                print(" ")


    return perimeter

def click_event(event:np.intc, x:np.intc, y:np.intc, flags, params)->None:
    '''
    Function to handle the mouse events
    Parameters:    event(intc) = mouse event 
                    x(intc) = x coordinate of the pixel
                    y(intc) = y coordinate of the pixel
                    flags,params not used

    Returns:       None
    '''
    global points_real, frame,data,z,middle_button_pressed

    if event == cv2.EVENT_MBUTTONDOWN:
        middle_button_pressed = True  
        if len(points_real) > 1 and len(points_real) != 2:  
            perimeter = compute_line_segments(points_real)
            print(f"The perimeter of the figure is: {perimeter}")
        if len(points_real) == 2:
            perimeter = compute_line_segments(points_real)
            perimeter = perimeter / 2
            print(f"The perimeter of the figure is: {perimeter}")

    if event == cv2.EVENT_LBUTTONDOWN:
        if middle_button_pressed:  
            return  
        centerX, centerY = frame.shape[1] // 2, frame.shape[0] // 2
        x_translated = x - centerX 
        y_translated =-(centerY - y)
        
        print(f"\nPixel coordinate measured with respect to image center:{x_translated}, {y_translated}") 
        points.append((x, y))
        calculate_real_coord(data,x,y,z)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        if len(points) >= 2:
            cv2.line(frame, points[-2], points[-1], (255, 0, 0), 2)
        
        if len(points) > 1:
            perimeter = compute_line_segments(points_real)

def open_camera(data:NDArray,index:np.intc)->None:
    '''
    Function to open camera and call calibration function, click events function and draw the lines between the points
    Parameters:     data(NDArray): Calibration data
                    index(int): camera index

    Returns:       None
    '''
    global points,points_real,distance_array,middle_button_pressed,frame,z

    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['distortion_coefficients'])

    cap = cv2.VideoCapture(index)
    cv2.namedWindow("Live video")
    cv2.setMouseCallback("Live video", click_event)

    if not cap.isOpened():
        print("Error: Could not found the camera.")
        sys.exit(0)
    else:

        while True:
            ret, frame = cap.read()
            frame = undistort_image(frame, camera_matrix, dist_coeffs)

            for pt in points: 
                cv2.circle(frame, pt, 3, (0, 0, 255), -1)

            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)

            if len(points) > 2:
                cv2.line(frame, points[-1], points[0], (255, 0, 0), 2)

            cv2.imshow('Live video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if keyboard.is_pressed('ctrl'):
                points=[]
                points_real=[]
                distance_array=[]
                middle_button_pressed=False
                time.sleep(0.1)
                print(f'Coordinates reseted')

        cap.release()
        cv2.destroyAllWindows()

def run_pipeline()->None:
    global data,z
    args = parser_user_data()
    calibration_data = args.cal_file
    data = load_calibration(calibration_data)
    index = args.cam_index
    z = args.Z
    open_camera(data,index)


if __name__ == '__main__':
    run_pipeline()