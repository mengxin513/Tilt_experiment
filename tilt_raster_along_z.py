from __future__ import print_function
import argparse
from openflexure_microscope import load_microscope
from contextlib import closing
import data_file
from openflexure_microscope.microscope import picamera_supports_lens_shading
import numpy as np
import time
import cv2
from scipy import ndimage

def measure_txy(ms, start_t, fraction):
    txy = np.zeros((1, 3))
    txy[0, 0] = time.time() - start_t
    frame = ms.rgb_image().astype(np.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame += (frame.max() - frame.min()) * fraction - frame.max()
    frame = cv2.threshold(frame, 0, 0, cv2.THRESH_TOZERO)[1]
    peak = ndimage.measurements.center_of_mass(frame)
    centre = (peak[1], peak[0])
    txy[0, 1:] = centre
    return txy

def tilt_scan_z(ms, z, step, start_t, backlash, fraction, experiment_group):

    def move_overlay(cx, cy, display_w, display_h):
        """move the overlay to show a shift of cx, cy camera pixels"""
        factor = display_h / 480.0
        x = int((display_w - 640.0 * factor) / 2.0 + cx * factor - 32.0 / 2.0 * factor)
        y = int(cy * factor - 32.0 / 2.0 * factor)
        overlay.window = (x, y, int(32 * factor), int(32 * factor))

    def move_stage(x, y, z, ms, start_t, fraction, standard_group):
        data_group = df.new_group('data', 'tilt_scan_along_z', parent = z_group)
        data_group['stage_position'] = stage.position
        txy = measure_txy(ms, start_t, fraction)
        move_overlay(txy[0, 1], txy[0, 2], 1920, 1080)
        data_group['cam_position'] = txy
        stage.move_rel([x, y, z])
        time.sleep(0.1)

    if backlash == 0:
        stage.backlash = False
    else:
        stage.backlash = backlash
    z_group = df.new_group('tilt_scan_along_z', 'tilt_experiment', parent = experiment_group)
    z_group.attrs['range'] = z
    z_group.attrs['step'] = step
    stage.move_rel([0, -z / 2, 0])
    for i in range(0, z, step):
            move_stage(0, step, 0, ms, start_t, fraction, z_group)
    data_group = df.new_group('data', 'tilt_scan_along_z', parent = z_group)
    data_group['stage_position'] = stage.position
    txy = measure_txy(ms, start_t, fraction)
    move_overlay(txy[0, 1], txy[0, 2], 1920, 1080)
    data_group['cam_position'] = txy
    stage.move_abs(initial_stage_position)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Tilt scan along z')
    parser.add_argument('z', type = int, help = 'Total scan distance along z in steps')
    parser.add_argument('step', type = int, help = 'Distance between points measured in steps')
    parser.add_argument('--backlash', type = int, default = 256, help = 'Backlash correction on or off')
    args = parser.parse_args()

    with load_microscope('microscope_settings.npz', dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = 'tilt_along_z.hdf5')) as df:

        assert picamera_supports_lens_shading(), 'You need the updated picamera module with lens shading!'

        camera = ms.camera
        stage = ms.stage

        z = args.z
        step = args.step
        backlash = args.backlash

        fraction = 0.3

        camera.resolution = (640, 480)
        camera.start_preview(resolution = (640, 480))

        stage.move_rel([-backlash, -backlash, -backlash])
        stage.move_rel([backlash, backlash, backlash])

        initial_stage_position = stage.position

        experiment_group = df.new_group('tilt', 'performes a tilt scan along z')

        pad = np.zeros((32, 32, 3), dtype = np.uint8)
        pad[:, 15:17, :] = 255
        pad[15:17, :, :] = 255
        pad[8:-8, 8:-8, :] = 0
        overlay = camera.add_overlay(pad.tobytes(), size = (32, 32), layer = 3, alpha = 128, fullscreen = False)

        start_t = time.time()

        tilt_scan_z(ms, z, step, start_t, backlash, fraction, experiment_group)

        camera.stop_preview()
        print('Done')
