'''
Can use this script to find the mapping between control and throttle
'''

import airsim
import numpy as np
from drone_util import get_throttle

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()
airsim.wait_key('Now we start our task!')

g = 9.81    # acc. due to gravity
t = 10       # total time you want to apply the control for (seconds)

## We have assumed throttle = kt * u4_bar. 
## We know that when u4_bar = g, the drone should hover
## therefore, you can visually check which value of kt makes the 
## drone hover in the simulation.
#####################
kt = 0.59375      # set this to a value that makes the drone hover
#####################

throttle = get_throttle(g,kt)
client.moveByAngleRatesThrottleAsync(0, 0, 0, min(throttle,1.0), t).join()

# disarm and safely exit
airsim.wait_key('Phew!')
client.armDisarm(False)
client.reset()
client.enableApiControl(False)