import sys
from threading import Thread
from pathlib import Path
from glob import glob
import time
from functools import reduce
import os
from ultralytics.yolo.v8.detect.predict import predict, getPredictor, set_data, init_dl, resetCounter
from qt.test import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QInputDialog
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal
import cv2
import numpy as np
from qt.label import Label

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.predictor = None
        
        self.predict_thread = None
        self.img = np.ndarray([])
        self.mode = 'train'
        self.running = False
        self.cap = None
        self.my_cfg = {'model': 'yolov8l.pt'}
        self.state = 'beginning'
        self.setting_line = False
        self.setting_file = False

    
    def initialize(self):
        if self.setting_line and self.setting_file:
            self.state = 'setting'
        if self.state != 'setting' and self.state != 'ready':
            # ret = QMessageBox.question(self, 'text', 'this is a text',  QMessageBox.Ok | QMessageBox.Cancel)
            # print(f'{ret = }, {QMessageBox.Ok}, {QMessageBox.Cancel}')
            QMessageBox.about(self, '提示', '模型未完成设置')
            return
        if self.predictor is not None:
            self.predictor = None
        set_data(self)
        predict()
        self.predictor = getPredictor()
        self.predict_thread = None
        self.state = 'ready'
        self.running = False
        cv2.destroyAllWindows()
        resetCounter()
        QMessageBox.about(self, '提示', '模型已成功初始化')

    def _Thread_Start(self):
        if self.predictor is not None and self.predict_thread is not None:
            init_dl(self)
            self.running = True
            self.predictor(data=self)

    def select_model(self):
        item, ok = QInputDialog.getItem(self, '选择模型', '选择你的预训练模型:', ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'])
        # print(f'{item = }, {ok = }')
        if ok:
            self.my_cfg['model'] = item + '.pt'

    def predict(self):
        if self.state != 'ready':
            QMessageBox.about(self, '警告', '模型尚未初始化')
            return
        self.mode = 'predict'
        try:
            self.predict_thread = Thread(target = self._Thread_Start)
            self.predict_thread.start()
        except RuntimeError:
            QMessageBox.about(self, '提示', '模型已运行结束')

    def train(self):
        if self.state != 'ready':
            QMessageBox.about(self, '警告', '模型尚未初始化')
            return
        self.mode = 'train'
        try:
            self.predict_thread = Thread(target = self._Thread_Start)
            self.predict_thread.start()
        except RuntimeError:
            QMessageBox.about(self, '提示', '模型已运行结束')

    def open_file(self):
        filePath, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./", "*.*")
        filePath = Path(filePath)
        self.stem = filePath.stem
        if filePath.suffix not in [".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv"]:
            QMessageBox.about(self, '提示', '请导入视频文件')
            return
        self.my_cfg['source'] = filePath

        _translate = QtCore.QCoreApplication.translate
        self.label_4.setText(str(filePath))

        self.cap = cv2.VideoCapture(str(filePath))
        if self.cap.isOpened():
            ret, self.img = self.cap.read()
            if not ret:
                QMessageBox.about(self, '提示', '视频读取失败')
                return
            height, width, channel = self.img.shape
            bytesPerline = 3 * width
            self.label_5.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            # 将QImage显示出来
            self.label_5.pixmap = QPixmap.fromImage(self.label_5.qImg)
            self.setting_file = True

    def save_file(self):
        if self.my_cfg.get('source', None) == None:
            QMessageBox.about(self, '提示', '数据未加载或模型未执行完成')
            return
        time.sleep(1)
        root = Path(__file__).parent / 'runs/detect'
        files = glob(str(root) + r'\**\*.*')
        # file_number = map(lambda a: int(Path(a).stem[5:] if Path(a).stem != 'train' else 0), files)
        file_number = []
        for i in files:
            file = Path(i)
            prefix = str(file.parts[-2])
            if prefix != 'train':
                file_number.append(int(prefix[5:]))
            else:
                file_number.append(0)
        max_file_number = max(file_number)
        QMessageBox.about(self, '提示', '视频已保存，位置为runs/detect/train' + str(max_file_number) if max_file_number != 0 else '' + '/'  + self.stem + '.mp4')


    def stop(self):
        self.running = False

    def draw_line(self):
        # if self.graphicsView.newPos and self.graphicsView.oldPos:
        #     self.graphicsView.drawLine(*self.graphicsView.oldPos, *self.graphicsView.newPos)
        self.label_5.removePos()
        self.line_set = False
        self.label_5.canPaint = True
        self.setting_line = True
        Thread(target=self.try_get_line).start()

    def try_get_line(self):
        while not self.label_5.line_set or self.label_5.canPaint or not self.label_5.oldPos or not self.label_5.newPos:
            time.sleep(0.1)
            pass
        self.my_cfg['line'] = [self.label_5.oldPos, self.label_5.newPos]
        print(self.my_cfg['line'])

    def waitUntilEnd(self):
        if self.predictor is not None:
            self.predict_thread.join()

    def resizeEvent(self, event): # retrive window size
        new_size = event.size()
        print("Window resized to:", new_size.width(), new_size.height())

    def closeEvent(self, event):
        if self.predict_thread and self.predict_thread.is_alive():
            QMessageBox.about(self,
                                        '提示',
                                        "程序正在运行，请等待运行结束或结束其运行。")
            event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    exit_code = app.exec_()
    mw.waitUntilEnd()