from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon, QMouseEvent,QPixmap,QPainter,QPen,QColor,QBrush,QFont
from PyQt5.QtCore import Qt,QRect

class Label(QLabel):
    def __init__(self, parent):
        super(Label, self).__init__()
        self.setStyleSheet("border: 2px solid red")
        self.parent = parent
        self.Pos = None
        self.oldPos = None
        self.newPos = None
        self.canPaint = False
        self.turn = 1
        self.pixmap = None
        self.setMouseTracking(True)
        self.line_set = False

    def mousePressEvent(self, event):
        if self.canPaint:
            self.Pos = event.pos()
            self.oldPos = self.Pos.x(), self.Pos.y()
            self.turn = 1
    
    def mouseMoveEvent(self, event):
        if self.canPaint:
            # print('moving...')
            if self.turn:
                self.Pos = event.pos()
                self.newPos = self.Pos.x(), self.Pos.y()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.canPaint:
            self.turn = 0
            self.canPaint = False
            self.line_set = True
    
    def removePos(self):
        self.oldPos = None
        self.newPos = None

    def paintEvent(self,event):
        # print('painting')
        if not self.pixmap:
            return
        
        Label_painter=QPainter(self)
        Label_painter.drawPixmap(0, 0, self.pixmap)

        if not self.canPaint:
            return
        
        self.pixmap = QPixmap.fromImage(self.qImg)
        painter=QPainter(self.pixmap)
        # painter.begin(self)
        painter.setPen(QPen(Qt.green, 3, Qt.SolidLine))
        if self.oldPos and self.newPos:
            painter.drawLine(*self.oldPos, *self.newPos)
        # self.painter.drawLine(50, 50, 100, 100)
        # painter.end()
