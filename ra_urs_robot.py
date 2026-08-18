import time
import serial
import threading
import numpy as np
import cv2
import re
class RaUrsRobot:
    def __init__(self):
        self.serial_port = '/dev/ttyACM0'
        self.laser_port = '/dev/ttyACM1'
        self.camera_id = '/dev/video0'
        self.combined_ser = None
        self.laser_ser = None
        self.cap = None
        self.joint_state = {'LINEAR': 0.0, 'ROTATION': 0.0, 'FLEXION': 0.0}
        self.joint_state_lock = threading.Lock()
        self._stop_encoders = threading.Event()
        self.encoder_thread = None
        self.re_linear = re.compile(r'Final Linear Relative Motion.*?:\s*([-+]?\d*\.\d+)')
        self.re_rotation = re.compile(r'Final Rotation Motor Deg, Ureteroscope Deg.*?:\s*[-+]?\d*\.\d+,\s*([-+]?\d*\.\d+)')
        self.re_flexion = re.compile(r'Final Flexion Step, Motor Deg, Tip Deg.*?:\s*[-+]?\d+,\s*[-+]?\d*\.\d+,\s*([-+]?\d*\.\d+)')

    def connect(self):
        try:
            self.combined_ser = serial.Serial(self.serial_port, 115200, timeout=0.1)
            self._stop_encoders.clear()
            self.encoder_thread = threading.Thread(target=self._read_serial, daemon=True)
            self.encoder_thread.start()
        except: pass
        try:
            self.laser_ser = serial.Serial(self.laser_port, 9600, timeout=0.1)
        except: pass
        self.cap = cv2.VideoCapture(self.camera_id)

    def _read_serial(self):
        while not self._stop_encoders.is_set():
            if self.combined_ser and self.combined_ser.in_waiting > 0:
                try:
                    line = self.combined_ser.readline().decode(errors='ignore').strip()
                    m1 = self.re_linear.search(line)
                    if m1:
                        with self.joint_state_lock: self.joint_state['LINEAR'] = float(m1.group(1))
                    m2 = self.re_rotation.search(line)
                    if m2:
                        with self.joint_state_lock: self.joint_state['ROTATION'] = float(m2.group(1))
                    m3 = self.re_flexion.search(line)
                    if m3:
                        with self.joint_state_lock: self.joint_state['FLEXION'] = float(m3.group(1))
                except: pass
            else:
                time.sleep(0.01)

    def capture_observation(self):
        obs = {}
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    frame_cropped = frame[180:1420, 220:1830]
                    frame_resized = cv2.resize(frame_cropped, (800, 800), interpolation=cv2.INTER_AREA)
                    frame_disp = cv2.rotate(frame_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    obs['observation.images.wolf_cam'] = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
                except: pass
        with self.joint_state_lock:
            obs['observation.state'] = np.array([self.joint_state['LINEAR'], self.joint_state['ROTATION'], self.joint_state['FLEXION'], 0.0], dtype=np.float32)
        return obs

    def send_action(self, action):
        target_d1 = np.clip(action[0], 0.0, 177.0)
        target_t1 = np.clip(action[1], -270.0, 270.0)
        target_t3 = np.clip(action[2], -270.0, 270.0)
        target_laser = action[3]
        with self.joint_state_lock:
            delta_d1 = target_d1 - self.joint_state['LINEAR']
            delta_t1 = target_t1 - self.joint_state['ROTATION']
        delta_t1_cmd = delta_t1 * -1.0
        if self.combined_ser:
            self.combined_ser.reset_input_buffer()
            self.combined_ser.write(f'{delta_d1:.4f} {delta_t1_cmd:.4f} {target_t3:.4f}\n'.encode())
        if self.laser_ser:
            self.laser_ser.write(b'f' if target_laser > 0.5 else b'p')

    def disconnect(self):
        self._stop_encoders.set()
        if self.encoder_thread: self.encoder_thread.join(timeout=1.0)
        if self.combined_ser: self.combined_ser.close()
        if self.laser_ser:
            try:
                self.laser_ser.write(b'x')
                time.sleep(0.1)
                self.laser_ser.close()
            except: pass
        if self.cap: self.cap.release()

if __name__ == '__main__':
    print('RA-URS LeRobot Node defined. Connection skipped to prevent dev port errors.')
