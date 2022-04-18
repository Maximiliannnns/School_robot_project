#!/usr/bin/env python
# Copyright (c) 2016, Universal Robots A/S,
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Universal Robots A/S nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL UNIVERSAL ROBOTS A/S BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
sys.path.append('..')
import logging

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import numpy as np
import cv2
import math
from pypylon import pylon
from threading import Thread

#logging.basicConfig(level=logging.INFO)

ROBOT_HOST = '192.168.1.18'
ROBOT_PORT = 30004
config_filename = 'control_loop_configuration.xml'

keep_running = True

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')
setp_names, setp_types = conf.get_recipe('setp')
watchdog_names, watchdog_types = conf.get_recipe('watchdog')

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

# get controller version
con.get_controller_version()

# setup recipes
con.send_output_setup(state_names, state_types)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

# Setpoints to move the robot to
setp1 = [-0.12, -0.43, 0.1, 0, 3.11, 0.04]
setp2 = [-0.12, -0.51, 0.1, 0, 3.11, 0.04]

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

# The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
watchdog.input_int_register_0 = 0

w_px = 750
w_mm = 440
h_px = 500
h_mm = 285
pixel_mm1 = (w_mm / w_px)
pixel_mm2 = (h_mm / h_px)

color_blue = (255, 0, 0)
color_yellow = (0, 255, 255)

#коэфеценты для исправления искажения изображения
dist_coef = np.array([[-1.51948828e+00,  1.95329695e+02, -1.04202010e-02, -3.03130271e-03,
  -7.54902603e+03]])
#camera_matrix = np.array([[640, 0, 480], [0, 640, 480], [0, 0, 1]])
camera_matrix = np.array([[3.32555825e+03, 0.00000000e+00, 9.49095102e+02],
 [0.00000000e+00, 3.20972678e+03, 5.61755191e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

setp1 = [0, 0, 0, 0, 0, 0]

def get_miny_point(box):
    ymin = box[0]
    for p in box:
        if p[1] < ymin[1]:
            ymin = p
    return ymin

def get_minx_point(box):
    xmin = box[0]
    for p in box:
        if p[0] < xmin[0]:
            xmin = p
    return xmin


def find_box_list(img_original):

    setp1 = [0, 0, 0, 0, 0, 0]  # default if countours0 = 0

    if img_original is None:
        return setp1

    hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)  # цвет меняю с BGR на HSV
    img_binary = cv2.threshold(hsv, 100, 255, cv2.THRESH_BINARY)[1]
    contours0, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours0:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат

        center = (int(rect[0][0]), int(rect[0][1]))
        # ищу координаты центра квадрата
        X = center[0]
        Y = center[1]
        # перевожу в мм
        Xm = (X * pixel_mm1) / 1000
        Ym = (Y * pixel_mm2) / 1000
        area = int(rect[1][0] * rect[1][1])  # вычисление площади

        if area > 20000 and area < 25000:
            pymin = get_miny_point(box)  # точка с  мин у среди всех вершин квадрата
            pxmin = get_minx_point(box)  # точка с  мин х среди всех вершин квадрата
            usedEdge = np.int0((pxmin[0] - pymin[0], pxmin[1] - pymin[1]))
            reference = (1, 0)  # горизонтальный вектор, задающий горизонт

            # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
            angle = 180 - (180.0 / math.pi * math.acos(
                (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (
                        cv2.norm(reference) * cv2.norm(usedEdge))))
            angle1 = (math.pi * angle) / 180

            setp1 = [Xm, Ym, 0.14, 0, 3.14, angle1]

            if area > 500:
                cv2.drawContours(img_original, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник
                cv2.circle(img_original, center, 5, color_yellow, 2)  # рисуем маленький кружок в центре прямоугольника
                # выводим в кадр величину угла наклона
                cv2.putText(img_original, "%d" % int(angle), (center[0] + 20, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)

    return img_original, setp1

def setp_to_list(setp):
    list = []
    for i in range(0,6):
        list.append(setp.__dict__["input_double_register_%i" % i])
    return list

def list_to_setp(setp, list):
    for i in range (0,6):
        setp.__dict__["input_double_register_%i" % i] = list[i]
    return setp


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

#start data synchronization
if not con.send_start():
    sys.exit()

while camera.IsGrabbing() and keep_running:

    state = con.receive()

    if state is None:
        break

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img_arr = image.GetArray()

        img_undist = cv2.undistort(img_arr, cameraMatrix=camera_matrix, distCoeffs=dist_coef)
        img_undist = img_undist[450:950, 550:1300]
        result_img, setp1 = find_box_list(img_undist)

        if state.output_int_register_0 != 0:
            list_to_setp(setp, setp1)
            print("send")
            # send new setpoint
            con.send(setp)

        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', result_img)
        k = cv2.waitKey(1)
        if k == 27:
            break

        # kick watchdog
        con.send(watchdog)

    grabResult.Release()

    # Releasing the resource
camera.StopGrabbing()

con.send_pause()

con.disconnect()
