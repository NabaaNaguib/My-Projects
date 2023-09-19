from pathlib import Path
import numpy as np
import argparse
import cv2
import copy
import depthai as dai
import torch
import serial
import time
from calc import HostSpatialsCalc
from utility import *
import math
import struct
    
def img_to_array(image):   
    if isinstance(image, np.ndarray):
        return image[:, :, ::-1]
    else: 
         array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
         array = np.reshape(array, (image.height, image.width, 4))
         array = array[:, :, :3]
         array = array[:, :, ::-1]
         return array

def dist_point_linestring(p, line_string):
    a = line_string[:-1, :]
    b = line_string[1:, :]
    return np.min(linesegment_distances(p, a, b))

def linesegment_distances(p, a, b):
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


def get_trajectory_from_lane_detector(ld, image):
    # get lane boundaries using the lane detector
    img = img_to_array(image)
    poly_left, poly_right, left_mask, right_mask = ld.get_fit_and_probs(img)
    # trajectory to follow is the mean of left and right lane boundary
    # according to our lane detector x is forward and y is left
    x = np.arange(-2,60,1.0)
    y = 0.5*(poly_left(x)+poly_right(x))
    # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
    # hence correct x coordinates
    #x += 0.5
    traj = np.stack((x,y)).T
    return traj, ld_detection_overlay(img, left_mask, right_mask)

def ld_detection_overlay(image, left_mask, right_mask):
    res = copy.copy(image)
    res[left_mask > 0.5, :] = [0,0,255]
    res[right_mask > 0.5, :] = [255,0,0]
    return res



# Initialize the serial port

try:
    ser = serial.Serial('COM3', 9600) # replace 'COM3' with the port of your Arduino board and 9600 with the baud rate of your choice
except serial.SerialException as e:
    print(f"Serial port error: {e}")
    ser = None


from Functions import CalibratedLaneDetector
from Functions import CameraGeometry    
from Functions import PurePursuit

PurePursuitController = PurePursuit()


def main(use_ADAS=False):
    # Imports
    global ser
    
    try:

        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(800, 400) # 1280, 720
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        # Create Oak-D output queues
        q_preview = pipeline.create(dai.node.XLinkOut)
        q_preview.setStreamName("preview")
        
        # Linking
        cam.preview.link(q_preview.input)
        
        # Define sources and outputs
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        stereo.initialConfig.setConfidenceThreshold(255)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        #stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("disp")
        stereo.disparity.link(xoutDepth.input)
        

        if use_ADAS:
            cg = CameraGeometry(image_width=800, image_height=400)
            # TODO: Change this line so that it works with your lane detector implementation
            ld = CalibratedLaneDetector(model_path=Path("./fastai_model.pth").absolute(), cam_geom=cg, calib_cut_v = 200)
            torch.cuda.is_available()

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)
            modelo = torch.hub.load('.', 'custom', 'yolov7-tiny.pt', source='local')
            
            # Connect to the Oak-D camera and start the pipeline
            with dai.Device(pipeline) as device:
                # Start the camera preview stream
                cam_preview = device.getOutputQueue("preview", 1, False)
                cam_preview.setBlocking(False)
                depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                dispQ = device.getOutputQueue(name="disp", maxSize=4, blocking=False)
                
                text = TextHelper()
                hostSpatials = HostSpatialsCalc(device)
                delta = 10
                hostSpatials.setDeltaRoi(delta)
                color = (255, 255, 255)
                max_error = 0
                
                # Create a timer object
                start_time = time.time()
                frame_count = 0
                fps = 0  # Initialize FPS to 0
                # Main loop
                while True:
                    # Read the latest camera preview frame
                    in_preview = cam_preview.get()
                    if in_preview is not None:
                    
                        frame = in_preview.getCvFrame()
                        results = modelo(frame)
                        c = results.pandas().xyxy[0]
                        listc=[]
                        listc = c.values.tolist()
                        #print(listc)
                        rows= len(listc)
                        #print(rows)

                        # Get disparity frame for nicer depth visualization
                        disp = dispQ.get().getFrame()
                        disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
                        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

                        depthFrame = depthQueue.get().getFrame()

                        # Convert the preview frame to an RGB image
                        image_rgb = cv2.cvtColor(in_preview.getCvFrame(), cv2.COLOR_BGR2RGB)
                        
                        # Start timer
                        if frame_count == 0:
                            start_time = time.time()
            
                        if use_ADAS:
                                                        
                            # Perform lane detection on the RGB image
                            traj, viz = get_trajectory_from_lane_detector(ld, image_rgb)
                            # Calculate the desired steering angle using the Pure Pursuit controller
                            waypoints = traj[:, :2]
                            
                            #### if your vehicle is traveling at 30 km/h, you can set speed = 8.33 (which is the speed in meters per second).
                            
                            speed = 7 # Replace with actual vehicle speed
                            steer_degrees = PurePursuitController.get_control(waypoints, speed) # Normalized Steering Angle
                            steering_angle_int = int(steer_degrees)
                        
                            print(f"The Steering Angle: {steering_angle_int} deg")
                            
                            
                            dist = dist_point_linestring(np.array([0,0]), traj)
                            cross_track_error = int(dist*100)
                            max_error = max(max_error, cross_track_error)
                            
                            
                            # Send the steering control to the Arduino board via the serial port
                            '''
                            if ser is not None:
                                try:
                                    ser.write(bytes([steering_angle_int]))
                                    #ser.write(int(steering_angle_int).encode()) # convert steer to a string and send it to the Arduino board
                                except Exception as e:
                                    print(f"Failed to send data to serial port: {e}")
                                    ser.close()
                                    # Close the serial port
                                    ser = None
                            '''
                            # Resize the output of results.render() to match the size of viz
                            op = np.squeeze(results.render())
                            op = cv2.resize(op, (viz.shape[1], viz.shape[0]))

                            # Blend the two images together using cv2.addWeighted()
                            alpha = 0.3
                            beta = 0.7
                            gamma = 0
                            combined_image = cv2.addWeighted(op, alpha, viz, beta, gamma)


                            i=0
                            min_distance = float('inf') # Initialize the minimum distance to infinity

                            height1 = frame.shape[0]
                            width1  = frame.shape[1]

                            height = depthFrame.shape[0]
                            width  = depthFrame.shape[1]
                            if len(listc) >0 :
                                while i < rows :
                                    xx1=(listc[i][0]/ width1)
                                    yy1=(listc[i][1]/ height1)
                                    xx2=(listc[i][2]/ width1)
                                    yy2=(listc[i][3]/ height1)

                                    x1=int(xx1*width)
                                    y1=int(yy1*height)
                                    x2=int(xx2*width)
                                    y2=int(yy2*height)

                                    # Calculate spatial coordiantes from depth frame
                                    spatials, centroid = hostSpatials.calc_spatials(depthFrame, (x1,y1,x2,y2)) # centroid == x/y in our case
                                    
                                     # Get the center of the frame
                                    center_x = width / 2
                                    center_y = height / 2

                                    # Threshold for object detection in the center of the frame
                                    threshold = 0.4 # 40% of the width of the frame

                                    # Check each detected object
                                    for bbox in listc:
                                        # Calculate the centroid coordinates
                                        centroid_x = (bbox[0] + bbox[2]) / 2
                                        centroid_y = (bbox[1] + bbox[3]) / 2
    
                                        # Calculate the distance between the centroid and the center of the frame
                                        distance_x = abs(centroid_x - center_x)
                                        distance_y = abs(centroid_y - center_y)    
                                        
                                        # Check if the object is in the center of the frame
                                        if distance_x < center_x * threshold and distance_y < center_y * threshold:
                                            # Extract depth values within the bounding box of the detected object
                                            depth_values = depthFrame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
                                            # Compute the median depth value as the distance
                                            object_distance = np.median(depth_values) / 1000 # in meters
                                              
                                    # Extract depth values within the bounding box of the detected object
                                    #depth_values = depthFrame[y1:y2, x1:x2]

                                    # Compute the median depth value as the distance
                                    #object_distance = np.median(depth_values) / 1000 # in meters
                                    
                                    if object_distance < min_distance:
                                        min_distance = object_distance # Update the minimum distance if a closer object is found 
                                    
                                    #print(spatials)
                                    text.rectangle(disp, (x1, y1), (x2, y2))
                                    text.putText(disp, "X: " + ("{:.1f}m".format(spatials['x']/1000) if not math.isnan(spatials['x']) else "--"), (x1 + 10, y1 + 20))
                                    text.putText(disp, "Y: " + ("{:.1f}m".format(spatials['y']/1000) if not math.isnan(spatials['y']) else "--"), (x1 + 10, y1 + 35))
                                    text.putText(disp, "Z: " + ("{:.1f}m".format(spatials['z']/1000) if not math.isnan(spatials['z']) else "--"), (x1 + 10, y1 + 50))
                                    i =i+1
                            
                            setpoint = 1.5 # the desired distance between the vehicle and the detected object
                            
                            # Use the minimum distance as the distance of the nearest object to the camera
                            object_distance = min_distance
                            
                            #print(f"Distance to object: {object_distance}")
                            
                            if ser is not None:
                                try:
                                    # Pack the values into a byte string
                                    data = struct.pack("if", steering_angle_int, object_distance)
                                    # Send the byte string over the serial port
                                    ser.write(data)
                                except Exception as e:
                                    print(f"Failed to send data to serial port: {e}")
                                    ser.close()
                                    # Close the serial port
                                    ser = None                            
                            
                            # Display distance text on the visualization
                            cv2.putText(combined_image, f"Distance to object: {object_distance}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    
                            if object_distance <= setpoint:
                                cv2.putText(combined_image, f"Too Close, Braking...", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(combined_image, f"Safe", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                        
                            
                            # Display FPS text on the visualization if fps has been assigned a value
                            if fps != 0:
                                cv2.putText(combined_image, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                            else:
                                cv2.putText(combined_image, "FPS: Computing...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                            # Display steer text on the visualization
                            cv2.putText(combined_image, f"Steer: {steering_angle_int:.2f} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)                            
                            # Display error text on the visualization
                            cv2.putText(combined_image, f"Error: {cross_track_error}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                            # Display max_error text on the visualization
                            cv2.putText(combined_image, f"Max Error: {max_error}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                            # Check if the car got out of its lane
                            #if abs(steer_degrees) > max_angle_degrees:
                             #   cv2.putText(viz, "WARNING: Vehicle is moving out of its lane", (160, 475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                                
                            # Show the frame
                            cv2.putText(disp, "fps: {:.2f}".format(fps), (5,  25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
                            cv2.imshow("depth", disp)
                            cv2.setUseOptimized(True)
                            winname = "Advanced Driver-Assistance System"
                            cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(winname, 1440, 720)
                            # Display the blended image
                            cv2.imshow(winname, combined_image)

                            key = cv2.waitKey(1)
                            if key == ord('q'):
                                cv2.destroyAllWindows
                                break                            
                            
                            # Increment frame count
                            frame_count += 1

                            # Calculate elapsed time since the timer was created
                            elapsed_time = time.time() - start_time

                            # Compute FPS every 10 frames
                            if frame_count % 10 == 0:
                                fps = frame_count / elapsed_time
                                print(f"FPS: {fps:.2f}")

                            # Reset the timer and frame count every second
                            if elapsed_time > 1.0:
                                start_time = time.time()
                                frame_count = 0
                            
                            # Exit the loop if the 'q' key is pressed
                            if cv2.waitKey(1) == ord('q'):
                                break
       

                  
    except Exception as e:
        print(f"An error occurred: {e}")                                    
            
    finally:                
   
        # Cleanup
        cv2.destroyAllWindows()
        if 'device' in locals() and device is not None:
            device.close()
        
        # Close the serial port
        if ser is not None:
            ser.close()  
        

    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs The Lane Keeping System.')
    parser.add_argument("--adas", action="store_true", help="Use reference trajectory from your LaneDetector class")
    args = parser.parse_args()

    try:
        main(use_ADAS= args.adas)

    except KeyboardInterrupt:
        print('\nCancelled by essam and bode. Bye!')
