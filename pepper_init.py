#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
motion.setExternalCollisionProtectionEnabled("Arms", False)
TOO IMPORTANT
'''

import time
from naoqi import ALProxy

PEPPER_IP = "192.168.0.103"
PEPPER_PORT = 9559

motion = ALProxy("ALMotion", PEPPER_IP, PEPPER_PORT)
life = ALProxy("ALAutonomousLife", PEPPER_IP, PEPPER_PORT)
posture = ALProxy("ALRobotPosture", PEPPER_IP, PEPPER_PORT)

try:
    # life.setState("disabled")
    motion.wakeUp()
    posture.goToPosture("StandInit", 0.5)
    motion.setAngles('HeadYaw', -0.55, 0.1)
    motion.setStiffnesses("Body", 1.0)
    motion.setStiffnesses("RArm", 1.0)
    motion.setExternalCollisionProtectionEnabled("Arms", False)
    motion.setMoveArmsEnabled(False, False)

    print("Pepper RArm stiffness is ON. Press Ctrl+C to release RArm.")

    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nCtrl+C detected.")

finally:
    print("Setting RArm stiffness to 0.0...")
    try:

        posture.goToPosture("StandInit", 0.5)
        motion.setStiffnesses("RArm", 0.0)
        print("RArm stiffness released.")
    except Exception as e:
        print("Failed to release RArm stiffness: %s" % e)