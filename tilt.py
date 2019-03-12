from openflexure_microscope import load_microscope
from contextlib import closing
import data_file
from openflexure_microscope.microscope import picamera_supports_lens_shading
import numpy as np
import queue
import threading
import time
import cv2
from scipy import ndimage

def image_capture(start_t, event, ms, q):
    while event.is_set():
        frame = ms.rgb_image().astype(np.float32)
        capture_t = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q.put(frame)
        tim = capture_t - start_t
        q.put(tim)
        #print('Number of itms in the queue: {}'.format(q.qsize()))

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz") as ms, \
         closing(data_file.Datafile(filename = "tilt.hdf5")) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera

        camera.resolution = (640, 480)
        
        N_frames = 1000
        fraction = 0.3

        tilt_data = df.new_group("data", "time, camerax, cameray")

        camera.start_preview(resolution = (640, 480))

        pad = np.zeros((32, 32, 3), dtype = np.uint8)
        pad[:, 15:17, :] = 255
        pad[15:17, :, :] = 255
        pad[8:-8, 8:-8, :] = 0
        overlay = camera.add_overlay(pad.tobytes(), size = (32, 32), layer = 3, alpha = 128, fullscreen = False)

        def move_overlay(cx, cy, display_w, display_h):
            """move the overlay to show a shift of cx, cy camera pixels"""
            factor = display_h / 480.0
            x = int((display_w - 640.0 * factor) / 2.0 + cx * factor - 32.0 / 2.0 * factor)
            y = int(cy * factor - 32.0 / 2.0 * factor)
            overlay.window = (x, y, int(32 * factor), int(32 * factor))

        q = queue.Queue()
        event = threading.Event()

        start_t = time.time()
        t = threading.Thread(target = image_capture, args = (start_t, event, ms, q), name = 'thread1')
        event.set()
        t.start()

        try:
            while event.is_set():
                if not q.empty():
                    data = np.zeros((N_frames, 3))
                    for i in range(N_frames):
                        frame = q.get()
                        tim = q.get()
                        data[i, 0] = tim
                        frame += (frame.max() - frame.min()) * fraction - frame.max()
                        frame = cv2.threshold(frame, 0, 0, cv2.THRESH_TOZERO)[1]
                        peak = ndimage.measurements.center_of_mass(frame)
                        centre = (peak[1], peak[0])
                        data[i, 1:] = centre
                        move_overlay(peak[1], peak[0], 1920, 1080)
                        print("Displacement: {} [px]".format(np.linalg.norm(data[i, 1:3] - data[0, 1:3])))
                    df.add_data(data, tilt_data, "data")
                else:
                    time.sleep(0.5)
                print("Looping")
            print("Done")
        except KeyboardInterrupt:
            event.clear()
            t.join()
            camera.stop_preview()
            print ("Got a keyboard interrupt, stopping")
            