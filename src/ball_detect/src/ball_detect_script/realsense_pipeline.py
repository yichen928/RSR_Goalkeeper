import pyrealsense2 as rs
import numpy as np
import os
import time
# Configure depth and color streams
class realsense_params:
    def __init__(self,ci,di,ex):
        self.color_intrinsics = ci
        self.depth_intrinsics = di
        self.extrinsics = ex
        self.color_inner_matirx=np.mat(np.array([[ci.fx,0,ci.ppx],[0,ci.fy,ci.ppy],[0,0,1]]))
        self.depth_inner_matrix=np.mat(np.array([[di.fx,0,di.ppx],[0,di.fy,di.ppy],[0,0,1]]))
        self.color_to_depth_rotation=np.mat(np.array(ex.rotation).reshape(3,3))##相机转换矩阵 旋转矩阵
        self.color_to_depth_translation=np.mat(np.array(ex.translation))###平移矩阵

class realsense_pipeline:

    def __init__(self) -> None:
        ctx = rs.context()
        for sensor in ctx.query_devices()[0].query_sensors():
            module_name = sensor.get_info(rs.camera_info.name)
            print(module_name)

            if (module_name == "Stereo Module"):
                depth_sensor = sensor
                depth_found = True
            elif (module_name == "RGB Camera"):
                color_sensor = sensor
                color_found = True

        if not (depth_found and color_found):
            print("Unable to find both stereo and color modules")

        depth_sensor.set_option(rs.option.frames_queue_size, 1)
        color_sensor.set_option(rs.option.frames_queue_size, 1)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device('843112072265')
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        # Start streaming
        self.start_pipeline()
        self.get_camera_params()
        align_to = rs.stream.color
        self.align = rs.align(align_to)
    
    def start_pipeline(self):
        self.pipeline.start(self.config)

    def get_camera_params(self):
        # sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        # sensor.set_option(rs.option.enable_auto_exposure, True)
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(self.depth_scale)
        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        depth_profile = depth.get_profile()
        color_profile = color.get_profile()
        print(depth_profile)
        print(color_profile)
        cvsprofile = rs.video_stream_profile(color_profile)
        dvsprofile = rs.video_stream_profile(depth_profile)
        color_intrin = cvsprofile.get_intrinsics()
        depth_intrin = dvsprofile.get_intrinsics()
        extrin = depth_profile.get_extrinsics_to(color_profile)
        # print(type(color_intrin))
        self.params=realsense_params(color_intrin,depth_intrin,extrin)
        # D435_para.refresh_mat()
        # print(D435_para.color_to_depth_translation)
        ###获取相机内参数并且保存为矩阵####
        # print(color_to_depth_rotation,color_to_depth_translation)

    def refresh_mat(self):
        self.frames = self.pipeline.wait_for_frames()
        
        aligned_frames = self.align.process(self.frames)
        self.depth = aligned_frames.get_depth_frame()
        self.color = aligned_frames.get_color_frame()
        hole_filling = rs.hole_filling_filter()
        self.depth = hole_filling.process(self.depth)
        self.depthmat=np.asanyarray(self.depth.get_data())
        self.colormat=np.asanyarray(self.color.get_data())

    def get_rgb_image(self):
        self.refresh_mat()
        return self.colormat

    def convert_actual_pos(self, pixel_pos):
        x = int((pixel_pos[0] + pixel_pos[1]) / 2)
        y = int((pixel_pos[2] + pixel_pos[3]) / 2)
        z = self.sample_region(x, y, 3)
        result = rs.rs2_deproject_pixel_to_point(self.params.color_intrinsics, [x, y], z) 
        result = np.array(result) / 1000
        return [result[2], -result[0], -result[1]]

    def sample_region(self, x, y, size):
        x_ = np.linspace(x-size//2, x+size//2, 3)
        y_ = np.linspace(y-size//2, y+size//2, 3)
        xv, yv = np.meshgrid(x_, y_)
        xv = np.clip(xv.flatten(), 0, 639)
        yv = np.clip(yv.flatten(), 0, 479)
        z = [self.depthmat[int(yv[i]),int(xv[i])] for i in range(9)]
        return min(z)