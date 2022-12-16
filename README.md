## Real-Sim-Real Robotic Goalkeeper

This repository is for EECS C206A Group 15 course project. Please follow the instructions. **You must have a mini-cheetah robot to successfully run this code.** This repository includes the pre-trained models to run the codes. Please contact us if you prefer to train the models by yourself.

The code should be copied to two servers, one with GPUs and the other with ROS. Please install `roslibpy` to connect the two servers with rosbridge.

+ **On the ROS server:**

  You should install `ROS` on this server. The code to connect with Cheetah is not included in this repository. We directly use the NUC server in Prof. Sreenath' laboratory.

  Firstly, you should connect with the mini-cheetah robot. 

  1. Please open the M/C on the robot in the order of on-off-on.
  2. Please open the RC transmitter, and ensure that the left-most and right-most controllers are on top.
  3. Check the connection of the Ethernet to the cheetah robot.

  + Terminal 1:

    Connect to the Cheetah robot.

    ```
    ssh user@10.0.0.44
    cd robot-software/build
    ./run_mc_2.sh ./mit_ctrl
    ```

  + Terminal 2:

    ```
    cd CheetahSoftware-FSM-RL/scripts
    python network_config.py
    cd ../mc-build
    ./sim/sim
    ```

    Change the right-most controllers on the transmitter from top to middle, then to the bottom, and change the left-most controller on the transmitter from top to middle (the robot would stand up).

  + Terminal 3:

    Run the rosbridge to connect the two servers. **This command should be run before running the ball detection, pose estimation, latent encoding, ball position transform, planner, and controller.**

    ```
    roslaunch rosbridge_server rosbridge_websocket.launch
    ```

  + Terminal 4:

    Run the ball position transform node to transform the ball position from the camera frame to the world frame.

    ```
    roslaunch ball_pipeline tf_publisher.launch
    ```

  + Terminal 5:

    Run the planner.

    ```
    cd src\rl_planning_pose
    python rl_planner_ros.py
    ```

  + Terminal 6:

    Run the controller.

    ```
    cd src\rl_control
    python rl_udp_control.py
    ```

+ **On the GPU server:**

  We use Python 3.8 with PyTorch 1.9.1, Torchvision 1.10.1. You should also install [HybrIK](https://github.com/Jeff-sjtu/HybrIK) following their instruction.

  + Terminal 1:

    Run the ball detection node. The checkpoint can be downloaded [here](https://drive.google.com/file/d/1PDblkLiSpYmFhYl6c81DXQ0jmxWOnZ4w/view?usp=share_link).

    ```
    cd src\ball_detect\src\ball_detect_script
    python ball_detect_node.py
    ```

  + Terminal 2:

    Run the pose estimation node. The checkpoint can be downloaded [here](https://drive.google.com/file/d/11mROKFAbEG1hapMV0et7xYpvTQEWMYvQ/view?usp=share_link).

    ```
    cd src\pose_estimate\src
    python pose_detect_node.py
    ```

  + Terminal 3:

    Run the human latent encoder.

    ```
    cd src\human_latent\src
    python human_latent_node.py
    ```

    
