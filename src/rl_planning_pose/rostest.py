from __future__ import print_function
import roslibpy

client = roslibpy.Ros(host='192.168.1.61', port=9090)
client.run()

listener = roslibpy.Topic(client, '/vrpn_client_node/ball/odometry/mocap', 'nav_msgs/Odometry')
listener.subscribe(lambda message: print('Heard talking: ',message["pose"]["pose"]["position"]["x"], message["pose"]["pose"]["position"]["y"], message["pose"]["pose"]["position"]["z"]))

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()