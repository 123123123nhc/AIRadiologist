# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'GUI6.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os
import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import torch
from matplotlib import image as mpimg
from torchvision.transforms import transforms
from model_VGG import VGGNet
from prediction import predict, segmentation
from segmentation_gpu import UNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Ui_Dialog(object):

    # set up the User interface
    def setupUi(self, Dialog):

        # load three models
        self.model1 = torch.load(r"Model/model1_cnn.pt")
        self.model2 = VGGNet().to(device)
        self.model2.load_state_dict(torch.load(r"Model/model1_vgg.pt"))
        self.model3 = VGGNet().to(device)
        self.model3.load_state_dict(torch.load(r"Model/model1_vgg_mask.pt"))
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        FIRST_OUT_CHANNELS = 16
        self.model_unet = UNet(n_channels=1, n_classes=2, first_out_channels= FIRST_OUT_CHANNELS, bilinear=False).to(device)
        self.model_unet.load_state_dict(torch.load(r'Model/unet.pt'))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        Dialog.setObjectName("Dialog")
        Dialog.resize(1520, 781)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        Dialog.setFont(font)
        self.label_folder = QtWidgets.QLabel(Dialog)
        self.label_folder.setGeometry(QtCore.QRect(30, 610, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_folder.setFont(font)
        self.label_folder.setStyleSheet("#label_folder{\n"
"    \n"
"    color: rgb(255, 255, 255);\n"
"}")
        self.label_folder.setObjectName("label_folder")
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(360, 60, 20, 721))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.qlabel_Input_image = QtWidgets.QLabel(Dialog)
        self.qlabel_Input_image.setGeometry(QtCore.QRect(429, 120, 512, 512))

        self.qlabel_Input_image.setStyleSheet("#qlabel_Input_image{\n"
"    background-iamge: url(\"file:backgound.png\")\n"
"}")

        self.qlabel_Input_image.setText("")
        self.qlabel_Input_image.setPixmap(QtGui.QPixmap("background.png"))
        self.qlabel_Input_image.setObjectName("qlabel_Input_image")
        self.qlabel_ouput_image = QtWidgets.QLabel(Dialog)
        self.qlabel_ouput_image.setGeometry(QtCore.QRect(960, 120, 512, 512))
        self.qlabel_ouput_image.setText("")
        self.qlabel_ouput_image.setObjectName("qlabel_ouput_image")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(0, 0, 1521, 61))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.frame.setFont(font)
        self.frame.setStyleSheet("#frame{\n"
"    background-color: rgb(196, 199, 212);\n"
"\n"
"\n"
"}\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_inputfiles = QtWidgets.QLabel(self.frame)
        self.label_inputfiles.setGeometry(QtCore.QRect(20, 20, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputfiles.setFont(font)
        self.label_inputfiles.setStyleSheet("#label_inputfiles{\n"
"    color: rgb(255, 255, 255)\n"
"}")
        self.label_inputfiles.setObjectName("label_inputfiles")
        self.pushButton_segmentation = QtWidgets.QPushButton(self.frame)
        self.pushButton_segmentation.setGeometry(QtCore.QRect(1080, 20, 141, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_segmentation.setFont(font)
        self.pushButton_segmentation.setStyleSheet("#pushButton_segmentation{\n"
"    background-color: rgb(139, 135, 132) ;\n"
"    color: rgb(255, 255, 255)    \n"
"}")
        self.pushButton_segmentation.setObjectName("pushButton_segmentation")
        self.pushButton_save = QtWidgets.QPushButton(self.frame)
        self.pushButton_save.setGeometry(QtCore.QRect(1410, 20, 101, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_save.setFont(font)
        self.pushButton_save.setStyleSheet("#pushButton_save{\n"
"    background-color: rgb(139, 135, 132) ;\n"
"    color: rgb(255, 255, 255)    \n"
"}")
        self.pushButton_save.setObjectName("pushButton_save")
        self.pushButton_classification = QtWidgets.QPushButton(self.frame)
        self.pushButton_classification.setGeometry(QtCore.QRect(1240, 20, 150, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_classification.setFont(font)
        self.pushButton_classification.setStyleSheet("#pushButton_classification\n"
"{\n"
"    background-color: rgb(139, 135, 132) ;\n"
"    color: rgb(255, 255, 255)    \n"
"}")
        self.pushButton_classification.setObjectName("pushButton_classification")
        self.pushButton_browse = QtWidgets.QPushButton(self.frame)
        self.pushButton_browse.setGeometry(QtCore.QRect(120, 20, 98, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_browse.setFont(font)
        self.pushButton_browse.setStyleSheet("#pushButton_browse{\n"
"    background-color: rgb(139, 135, 132) ;\n"
"    color: rgb(255, 255, 255)    \n"
"}")
        self.pushButton_browse.setObjectName("pushButton_browse")
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setGeometry(QtCore.QRect(370, 660, 1221, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_diagnosis = QtWidgets.QLabel(Dialog)
        self.label_diagnosis.setGeometry(QtCore.QRect(390, 680, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_diagnosis.setFont(font)
        self.label_diagnosis.setObjectName("label_diagnosis")
        self.pushButton_lastpicture = QtWidgets.QPushButton(Dialog)
        self.pushButton_lastpicture.setGeometry(QtCore.QRect(430, 90, 121, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_lastpicture.setFont(font)
        self.pushButton_lastpicture.setStyleSheet("background-color: rgb(139, 135, 132) ;\n"
"    color: rgb(255, 255, 255)    ")
        self.pushButton_lastpicture.setObjectName("pushButton_lastpicture")
        self.pushButton_nextpricture = QtWidgets.QPushButton(Dialog)
        self.pushButton_nextpricture.setGeometry(QtCore.QRect(810, 90, 131, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_nextpricture.setFont(font)
        self.pushButton_nextpricture.setStyleSheet("background-color: rgb(139, 135, 132) ;\n"
"    color: rgb(255, 255, 255)    ")
        self.pushButton_nextpricture.setObjectName("pushButton_nextpricture")
        self.listView_file = QtWidgets.QListView(Dialog)
        self.listView_file.setGeometry(QtCore.QRect(30, 111, 311, 641))
        self.listView_file.setObjectName("listView_file")
        self.label_file = QtWidgets.QLabel(Dialog)
        self.label_file.setGeometry(QtCore.QRect(600, 90, 210, 20))
        self.label_file.setText("")
        self.label_file.setObjectName("label_file")
        self.label_diagonosis2 = QtWidgets.QLabel(Dialog)
        self.label_diagonosis2.setGeometry(QtCore.QRect(510, 680, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.textEdit_diagnosis = QtWidgets.QTextEdit(Dialog)
        self.textEdit_diagnosis.setGeometry(QtCore.QRect(500, 680, 431, 91))
        self.textEdit_diagnosis.setObjectName("textEdit_diagnosis")

        self.retranslateUi(Dialog)

        self.pushButton_browse.clicked.connect(self.openDirectory)
        self.pushButton_nextpricture.clicked.connect(self.next_picture)
        self.pushButton_lastpicture.clicked.connect(self.last_picture)
        self.pushButton_save.clicked.connect(self.saveImage)
        self.pushButton_segmentation.clicked.connect(self.getSegmentation)
        self.listView_file.clicked.connect(self.changeFile)
        self.pushButton_classification.clicked.connect(self.getClass)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_folder.setText(_translate("Dialog", "Folder"))
        self.label_inputfiles.setText(_translate("Dialog", "Input Folder:"))
        self.pushButton_segmentation.setText(_translate("Dialog", "Segmentation"))
        self.pushButton_save.setText(_translate("Dialog", "Save"))
        self.pushButton_classification.setText(_translate("Dialog", "Classification"))
        self.pushButton_browse.setText(_translate("Dialog", "Browse"))
        self.label_diagnosis.setText(_translate("Dialog", "Diagnosis:"))
        self.pushButton_lastpicture.setText(_translate("Dialog", "Last Picture"))
        self.pushButton_nextpricture.setText(_translate("Dialog", "Next Picture"))


    def openDirectory(self):
        '''

        :return: upload the folder from users' selection and show the first pictures in the folder in the left qlabel!
        '''

        self.fd = QFileDialog.getExistingDirectory(self.centralwidget, "choose the folder",
                                                           r"test")

        self.workingfiles = []
        for root, dirs, files in os.walk(self.fd):
            for file in files:
                if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':
                    self.workingfiles.append(file)
        if len(self.workingfiles) == 0:
            # warning for no picture!
            QMessageBox.warning(None,"Warning","No picture in the folder, please input png or jpg file!")
        else:
            self.listModel = QStringListModel()
            self.listModel.setStringList(self.workingfiles)
            self.listView_file.setModel(self.listModel)
            self.label_file.setText((self.workingfiles[0]))
            # resize the input image to (512,512). It is useful to show the pictures in the interface!
            path = os.path.join(self.fd, self.workingfiles[0])
            img = Image.open(path)
            img = img.resize((512, 512))
            out_path = os.path.join(r'resized_picture', self.workingfiles[0])
            img.save(out_path)
            self.qlabel_Input_image.setPixmap(QtGui.QPixmap(out_path))

    def changeFile(self,item):
        '''
        :param item:  item in the listView
        :return: change the file from list view
        '''
        #
        self.textEdit_diagnosis.clear()
        self.label_file.setText(self.workingfiles[item.row()])
        self.path_picture = os.path.join(self.fd, self.label_file.text())

        img = Image.open(self.path_picture)
        img = img.resize((512,512))
        out_path = os.path.join(r'resized_picture',self.label_file.text())
        img.save(out_path)
        # clear the input image label and diagnosis textEdit when the file is changed
        self.qlabel_Input_image.clear()
        self.qlabel_Input_image.setPixmap(QtGui.QPixmap(out_path))
        self.label_diagonosis2.clear()
        self.qlabel_ouput_image.clear()

    def next_picture(self):

        '''
        :return: find whether the folder is input
        '''

        if self.label_file.text() == "":
            QMessageBox.warning(None, 'Warning', 'Please input the folder first!')
        # To find whether the picture is the last picture
        elif self.workingfiles.index(self.label_file.text()) +1  >= len(self.workingfiles):
            QMessageBox.warning(None, 'Warning','It the last picture!')

        else:
            # get the path of next picture
            self.path_picture = os.path.join(self.fd, self.label_file.text())
            file_name = self.workingfiles[self.workingfiles.index(self.label_file.text()) + 1]
            path_next_picture = os.path.join(self.fd, file_name)
            # output the next picture
            self.label_file.setText((file_name))
            self.qlabel_Input_image.setPixmap(QtGui.QPixmap(path_next_picture))
            self.label_diagonosis2.clear()

    def last_picture(self):
        '''
        :return: find whether the folder is input
        '''

        # warning for no folders
        if self.label_file.text() == "":
            QMessageBox.warning(None, 'Warning', 'Please input the folder first!')
        # warning for no left picture
        elif self.workingfiles.index(self.label_file.text()) <= 0 :
            QMessageBox.warning(None, 'Warning','It the first picture!')
        else:
            self.path_picture = os.path.join(self.fd, self.label_file.text())
            file_name = self.workingfiles[self.workingfiles.index(self.label_file.text()) - 1]
            path_next_picture = os.path.join(self.fd, file_name)
            self.label_file.setText((file_name))
            self.qlabel_Input_image.setPixmap(QtGui.QPixmap(path_next_picture))
            self.label_diagonosis2.clear()

    # Save the image
    def saveImage(self):
        '''
        :return: save the mask
        '''

        self.path_picture_mask = os.path.join(r'img_mask', self.label_file.text())
        if os.path.exists(self.path_picture_mask):
            screen = QApplication.primaryScreen()
            pix = screen.grabWindow(self.qlabel_ouput_image.winId())
            fd, type = QFileDialog.getSaveFileName(self.centralwidget, "Save the picture",
                                                   r"",
                                                   "*.png;;All Files(*)")
            pix.save(fd)
        else:
            # warning for no segmentation
            QMessageBox.warning(None, 'Warning', 'Please segment first!')

    def getSegmentation(self):
        '''
        :return: the mask image and show in the right label
        '''

        self.path_picture = os.path.join(self.fd, self.label_file.text())
        img = Image.open(self.path_picture)
        # resize the input image to the size suitable to unet
        RESCALE_SIZE = (256,256)
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        transforms.Resize(RESCALE_SIZE)])
        img = transform(img) / 255
        img = img.to(device)
        img = torch.unsqueeze(img, dim=0)
        out = segmentation(img, self.model_unet, device)
        output_path = os.path.join(r'resized_picture', self.label_file.text())
        # masks can be saved in the resized_picture folder, which can be used to classify the picture in model3
        out.save(output_path)

        # produce the pictures covered by masks
        img = mpimg.imread(self.path_picture)
        img.flags.writeable = True
        mask = mpimg.imread(output_path)
        Image.fromarray(np.uint8(img))
        img[:, :][mask[:, :] == 0] = 255
        img[:, :][mask[:, :] == 1] *= 255
        img_mask_path = os.path.join('img_mask',self.label_file.text())
        cv2.imwrite((img_mask_path), img)

        self.qlabel_ouput_image.setPixmap(QtGui.QPixmap(img_mask_path))

    def getClass(self):
        '''

        :return: categories predicted by three models
        '''
        self.textEdit_diagnosis.clear()

        self.path_picture = os.path.join(self.fd, self.label_file.text())
        img = Image.open(self.path_picture)

        # rescale size for VGG
        img_size1 = (512, 512)
        # rescale size for scnn
        img_size2 = (224, 224)
        # transform to sCNN format
        transform1 = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Resize(img_size1),
                                        transforms.Normalize([0.5], [0.5])])
        # transform to VGG format
        transform2 = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Resize(img_size2),
                                        transforms.Normalize([0.5], [0.5])])

        self.path_picture = os.path.join(self.fd, self.label_file.text())
        self.path_picture_mask = os.path.join(r'img_mask', self.label_file.text())
        img1 = transform1(img)
        img1 = img1.unsqueeze(0)
        img2 = transform2(img)
        img2 = img2.unsqueeze(0)
        ans1 = predict(img1, self.model1, device) # prediction by model1
        ans2 = predict(img2, self.model2, device) # prediction by model2

        # 0: normal, 1: tuberculosis
        if ans1 == 0:
            self.textEdit_diagnosis.setText("Model1: Normal")
        if ans1 == 1:
            self.textEdit_diagnosis.setText("Model1: Tuberculosis")

        if ans2 == 0:
            self.textEdit_diagnosis.append("Model2: Normal")
        if ans2 == 1:
            self.textEdit_diagnosis.append("Model2: Tuberculosis")

        # whether the picture was covered by the mask
        if os.path.exists(self.path_picture_mask):
            img3 = Image.open(self.path_picture_mask)
            img3 = transform2(img3)
            img3 = img3.unsqueeze(0)
            ans3 = predict(img3, self.model3, device)  # prediction by model3
            if ans3 == 0:
                self.textEdit_diagnosis.append("Model3: Normal")
            if ans3 == 1:
                self.textEdit_diagnosis.append("Model3: Tuberculosis")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle("AIRadiologist")
    MainWindow.show()
    sys.exit(app.exec_())

