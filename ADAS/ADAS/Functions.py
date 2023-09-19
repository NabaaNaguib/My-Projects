import numpy as np
import cv2
import torch


def get_intrinsic_matrix(field_of_view_deg, image_width, image_height):
    # For our Carla camera alpha_u = alpha_v = alpha
    # alpha can be computed given the cameras field of view via
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu],
                     [0, alpha, Cv],
                     [0, 0, 1.0]])

def project_polyline(polyline_world, trafo_world_to_cam, K):
    x,y,z = polyline_world[:,0], polyline_world[:,1], polyline_world[:,2]
    homvec = np.stack((x,y,z,np.ones_like(x)))
    proj_mat = K @ trafo_world_to_cam[:3,:]
    pl_uv_cam = (proj_mat @ homvec).T
    u = pl_uv_cam[:,0] / pl_uv_cam[:,2]
    v = pl_uv_cam[:,1] / pl_uv_cam[:,2]
    return np.stack((u,v)).T


class CameraGeometry(object):
    def __init__(self, height=1.3, yaw_deg=0, pitch_deg=-5, roll_deg=0, image_width=1024, image_height=512, field_of_view_deg=68.7938):
        # scalar constants
        self.height = height
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.yaw_deg = yaw_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg
        # camera intriniscs and extrinsics
        self.intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, image_width, image_height)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                            [cp*sr, cp*cr, sp],
                                            [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
        self.rotation_cam_to_road = rotation_road_to_cam.T # for rotation matrices, taking the transpose is the same as inversion
        self.translation_cam_to_road = np.array([0,-self.height,0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3,0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3,3] = self.translation_cam_to_road
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])


    def camframe_to_roadframe(self,vec_in_cam_frame):
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_camframe(self,u,v):
        # NOTE: The results depend very much on the pitch angle (0.5 degree error yields bad result)
        # Here is a paper on vehicle pitch estimation:
        # https://refubium.fu-berlin.de/handle/fub188/26792
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height*Kinv_uv_hom/denominator
    
    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y]) # read book section on coordinate systems to understand this

    def precompute_grid(self,dist=60):
        cut_v = int(self.compute_minimum_v(dist=dist)+1)
        xy = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                X,Y,Z= self.uv_to_roadXYZ_roadframe_iso8855(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):    
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v


class LaneDetector():
    def __init__(self, cam_geom=CameraGeometry(), model_path='./fastai_model.pth'):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = torch.load(model_path).to(self.device)
        else:
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
            self.device = "cpu"
        self.model.eval()

    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image   

    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def _predict(self, img):
        with torch.no_grad():
            image_tensor = img.transpose(2,0,1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
            model_output = torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output

    def detect(self, img_array):
        model_output = self._predict(img_array)
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:] 
        return background, left, right
    
    def fit_poly(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        if mask.sum() > 0:
            coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        else:
            coeffs = np.array([0.,0.,0.,0.])
        return np.poly1d(coeffs)


    def __call__(self, image):
        if isinstance(image, str):
            image = self.read_imagefile_to_array(image)
        left_poly, right_poly, _, _ = self.get_fit_and_probs(image)
        return left_poly, right_poly

    def get_fit_and_probs(self, img):
        _, left, right = self.detect(img)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right
    
def get_intersection(line1, line2):
    m1, c1 = line1
    m2, c2 = line2
    if m1 == m2:
        return None
    u_i = (c2 - c1) / (m1 - m2)
    v_i = m1*u_i + c1
    return u_i, v_i

def get_py_from_vp(u_i, v_i, K):
    p_infinity = np.array([u_i, v_i, 1])
    K_inv = np.linalg.inv(K)
    r3 = K_inv @ p_infinity    
    r3 /= np.linalg.norm(r3)
    yaw = -np.arctan2(r3[0], r3[2])
    pitch = np.arcsin(r3[1])    
    
    return pitch, yaw

class CalibratedLaneDetector(LaneDetector):
    def __init__(self, calib_cut_v = 200, cam_geom=CameraGeometry(), model_path='./fastai_model.pth'):
        # call parent class constructor
        super().__init__(cam_geom, model_path)

        self.calib_cut_v = calib_cut_v

        self.estimated_pitch_deg = 0
        self.estimated_yaw_deg = 0
        self.mean_residuals_thresh = 15
        self.update_cam_geometry()
        self.pitch_yaw_history = []
        self.calibration_success = False

    def get_fit_and_probs(self, image):
        _, left_probs, right_probs = self.detect(image)
        line_left  = self._fit_line_v_of_u(left_probs)
        line_right = self._fit_line_v_of_u(right_probs)
        if (line_left is not None) and (line_right is not None):
            vanishing_point = get_intersection(line_left, line_right)
            if vanishing_point is not None:                
                u_i, v_i = vanishing_point
                pitch, yaw = get_py_from_vp(u_i, v_i, self.cg.intrinsic_matrix)
                self.add_to_pitch_yaw_history(pitch, yaw)

        left_poly = self.fit_poly(left_probs)
        right_poly = self.fit_poly(right_probs)
        return left_poly, right_poly, left_probs, right_probs
    
    def _fit_line_v_of_u(self, probs):
        v_list, u_list = np.nonzero(probs > 0.3)
        if v_list.size == 0:
            return None
        coeffs, residuals, _, _, _ = np.polyfit(
            u_list, v_list, deg=1, full=True)
            
        mean_residuals = residuals/len(u_list)
        #print(mean_residuals)
        if mean_residuals > self.mean_residuals_thresh:
            return None
        else:
            return np.poly1d(coeffs)

    def add_to_pitch_yaw_history(self, pitch, yaw):
        self.pitch_yaw_history.append([pitch, yaw])
        if len(self.pitch_yaw_history) > 50:
            py = np.array(self.pitch_yaw_history)
            mean_pitch = np.mean(py[:,0])
            mean_yaw = np.mean(py[:,1])
            self.estimated_pitch_deg = np.rad2deg(mean_pitch)
            self.estimated_yaw_deg = np.rad2deg(mean_yaw)
            self.update_cam_geometry()
            self.calibration_success = True
            self.pitch_yaw_history = []
            print("yaw, pitch = ", self.estimated_yaw_deg, self.estimated_pitch_deg)

    def update_cam_geometry(self):
        self.cg = CameraGeometry(
            height = self.cg.height, 
            roll_deg = self.cg.roll_deg,
            image_width = self.cg.image_width,
            image_height = self.cg.image_height, 
            field_of_view_deg = self.cg.field_of_view_deg,
            pitch_deg = self.estimated_pitch_deg, 
            yaw_deg = self.estimated_yaw_deg )
        self.cut_v, self.grid = self.cg.precompute_grid()


def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

def get_target_point(lookahead, polyline):
    intersections = []
    for j in range(len(polyline)-1):
        pt1 = polyline[j]
        pt2 = polyline[j+1]
        intersections += circle_line_segment_intersection((0,0), lookahead, pt1, pt2, full_line=False)
    filtered = [p for p in intersections if p[0]>0]
    if len(filtered)==0:
        return None
    return filtered[0]


# TODO: Tune parameters of PID with these global variables
param_Kp = 2
param_Ki = 0
param_Kd = 0
# TODO: Tune parameters of Pure Pursuit with these global variables
param_K_dd = 0.4
# The above parameters will be used in the Carla simulation
# The simple simulation in tests/control/control.ipynb does not use these parameters
    
class PurePursuit:
    def __init__(self, K_dd=param_K_dd, wheel_base=2.65, waypoint_shift=1.4):
        self.K_dd = K_dd
        self.wheel_base = wheel_base
        self.waypoint_shift = waypoint_shift
    
    def get_control(self, waypoints, speed):
        # transform x coordinates of waypoints such that coordinate origin is in rear wheel
        waypoints[:,0] += self.waypoint_shift
        look_ahead_distance = np.clip(self.K_dd * speed, 3,20)

        track_point = get_target_point(look_ahead_distance, waypoints)
        if track_point is None:
            return 0

        alpha = np.arctan2(track_point[1], track_point[0])

        # Change the steer output with the lateral controller.
        steer = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)

        # Convert the steering angle to degrees
        steering_angle_degrees = steer * (180/np.pi)

        # undo transform to waypoints 
        waypoints[:,0] -= self.waypoint_shift
        return steering_angle_degrees



class PIDController:
    def __init__(self, Kp, Ki, Kd, set_point):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.int_term = 0
        self.derivative_term = 0
        self.last_error = None
    
    def get_control(self, measurement, dt):
        error = self.set_point - measurement
        self.int_term += error*self.Ki*dt
        if self.last_error is not None:
            self.derivative_term = (error-self.last_error)/dt*self.Kd
        self.last_error = error
        return self.Kp * error + self.int_term + self.derivative_term