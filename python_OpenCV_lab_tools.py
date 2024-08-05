from customtkinter import *
import cv2
from PIL import Image
import numpy as np
import threading
import colorsys
from datetime import datetime
import easyocr

class ThreadCamera(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.debug = False
                
        self.cameraList = []
        self.cameraNumber = 0
        self.width = 2592
        self.height = 1944
        self.videoCapture = None
        
        self.image = None
        self.showImage = None
        
        self.isPerspectiveTransform = False
        self.isPerspectivePointSet = False
        
        self.isGray = False
        self.isBlur = False
        self.isBlur_gussian = False
        self.isMask_hsv = False
        self.isMask_gray = False
        self.isErosion = False
        self.isDilation = False
        self.isMorphOpen = False
        self.isMorphClose = False
        self.isMorphGradient = False
        
        self.imageProcessingOrder = []
        self.imageProcessingOrderName = []
        
        self.rotateAngle = 0
        
        self.perspective_transform_pointTL = [0,0]
        self.perspective_transform_pointBL = [0,0]
        self.perspective_transform_pointBR = [0,0]
        self.perspective_transform_pointBT = [0,0]
        
        self.blur_kernel_x = 1
        self.blur_kernel_y = 1
        
        self.blur_gussian_kernel_x = 1
        self.blur_gussian_kernel_y = 1
        
        self.mask_hsv_lower = [0, 0, 0]
        self.mask_hsv_upper = [0, 0, 255]
        
        self.mask_gray_lower = 0
        self.mask_gray_upper = 255
        
        self.erosion_kernel_x = 1
        self.erosion_kernel_y = 1
        
        self.dilation_kernel_x = 1
        self.dilation_kernel_y = 1
        
        self.morph_open_kernel_x = 1
        self.morph_open_kernel_y = 1
        
        self.morph_close_kernel_x = 1
        self.morph_close_kernel_y = 1
        
        self.morph_gradient_kernel_x = 1
        self.morph_gradient_kernel_y = 1
        
        self.ocrResult = ""
        
        self.cameraSetup()
        
    def run(self):
        while not self._stop_event.is_set():
            try:
                self.ret, self.frame = self.videoCapture.read()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            except:
                pass
            
            try:
                image = ImageProcessing.rotate_image(self.frame, self.rotateAngle)
                
                if self.isPerspectiveTransform:
                    if self.isPerspectivePointSet == False:
                        self.savePerspectiveTransformPoint()
                        # if Cancel will disable persective
                        if (self.isPerspectivePointSet == False):
                            self.isPerspectiveTransform = False
                    else:
                        image = ImageProcessing.perspectiveTransform(image, self.perspective_transform_pointTL, 
                                                                    self.perspective_transform_pointBL, 
                                                                    self.perspective_transform_pointBR, 
                                                                    self.perspective_transform_pointBT)
                
                image = self.imageProcessing(image)
            
                self.showImage = cv2.resize(image,(0,0),fx = 0.4, fy = 0.4)
                self.image = image
                
            except Exception as error:
                pass
        print("Camera thread stop")
    
    def imageProcessing(self, image):
        if len(self.imageProcessingOrder) > 0:
            for func in self.imageProcessingOrder:
                image = func(image)
        return image
    
    def add_imageProcessing(self, process):
        if process in self.imageProcessingOrder:
            return
        
        self.imageProcessingOrder.append(process)
        self.imageProcessingOrderName.append(process.__name__)
        
        print(self.imageProcessingOrderName)
    
    def remove_imageProcessing(self, process):
        if process not in self.imageProcessingOrder:
            return
        
        self.imageProcessingOrder.remove(process)
        self.imageProcessingOrderName.remove(process.__name__)
        
        print(self.imageProcessingOrderName)
    
    def imageProcessing_transform_perspective(self, image):
        if self.isPerspectivePointSet == False:
            self.savePerspectiveTransformPoint()
            # if Cancel will disable persective
            if (self.isPerspectivePointSet == False):
                self.isPerspectiveTransform = False
        else:
            image = ImageProcessing.perspectiveTransform(image, self.perspective_transform_pointTL, 
                                                        self.perspective_transform_pointBL, 
                                                        self.perspective_transform_pointBR, 
                                                        self.perspective_transform_pointBT)
                        
    def imageProcessing_grayscale(self, image):
        return ImageProcessing.grayscale(image)
    
    def imageProcessing_blur_gaussian(self, image):
        return ImageProcessing.gaussianBlur(image, self.blur_gussian_kernel_x, self.blur_gussian_kernel_y)
    
    def imageProcessing_mask_hsv(self, image):
        return ImageProcessing.hsvMask(image, self.mask_hsv_lower, self.mask_hsv_upper)
    
    def imageProcessing_mask_gray(self, image):
        return ImageProcessing.rangeMask(image, self.mask_gray_lower, self.mask_gray_upper)
    
    def imageProcessing_erosion(self, image):
        return ImageProcessing.erosion(image, self.erosion_kernel_x, self.erosion_kernel_y)
    
    def imageProcessing_dilation(self, image):
        return ImageProcessing.dilation(image, self.dilation_kernel_x, self.erosion_kernel_y)
    
    def imageProcessing_morph_open(self, image):
        return ImageProcessing.morphOpen(image, self.morph_open_kernel_x, self.morph_open_kernel_y)
    
    def imageProcessing_morph_close(self, image):
        return ImageProcessing.morphClose(image, self.morph_close_kernel_x, self.morph_close_kernel_y)
    
    def imageProcessing_morph_gradient(self, image):
        return ImageProcessing.morphGradient(image, self.morph_gradient_kernel_x, self.morph_gradient_kernel_y)
    
    def stop(self):
        self._stop_event.set()
    
    def cameraSetup(self):
        self.videoCapture = cv2.VideoCapture(self.cameraNumber)
        self.videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    
    def changeCameraNumber(self, cameraNumber):
        if cameraNumber != self.cameraNumber:
            self.cameraNumber = cameraNumber
            self.videoCapture.release()
            self.videoCapture = cv2.VideoCapture(self.cameraNumber)
            self.videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
    def snapshotImage(self):
        snapshot = Image.fromarray(np.array(self.frame))
        try:
            snapshot = Image.fromarray(np.array(self.frame))
            now = datetime.now()
            time = now.strftime("%H%M%S")
            date = now.strftime("%d%m%Y")
            file = f"Snapshot_{date}_{time}"
            imageType = ".bmp"
            
            if not os.path.isdir("Snapshot"):
                os.mkdir("Snapshot")
                
            snapshot.save(f"Snapshot/{file}{imageType}")
            
            print(f"Snapshot : {file}{imageType}")
        except:
            pass
        
    def snapshotProcessImage(self):
        try:
            snapshot = Image.fromarray(np.array(self.image))
            now = datetime.now()
            time = now.strftime("%H%M%S")
            date = now.strftime("%d%m%Y")
            file = f"PostProcess_Snapshot_{date}_{time}"
            imageType = ".bmp"
            
            if not os.path.isdir("Snapshot"):
                os.mkdir("Snapshot")
            
            snapshot.save(f"Snapshot/{file}{imageType}")
            
            print(f"Snapshot : {file}{imageType}")
        except:
            pass
    
    def savePerspectiveTransformPoint(self):
        if self.isPerspectivePointSet:
            return
        
        global pointTL, pointBL, pointBR, pointBT, mousePos, processImage, pointState, color, circleRadian, selectionPoint
        selectionPoint = []
        pointTL = False 
        pointBL = False
        pointBR = False
        pointBT = False
        pointState = "Point_TopLeft"
        
        mousePos = (0,0)
        color = (0, 0, 255)
        circleRadian = 5
        
        image = self.image.copy()
        image = cv2.resize(image,(0,0),fx = 0.5, fy = 0.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processImage = image.copy()
        
        def selectTransformPoint(event, x, y, flags, param):
            global pointTL, pointBL, pointBR, pointBT, mousePos, processImage, pointState, color, circleRadian, selectionPoint
            
            if event == cv2.EVENT_MOUSEMOVE:
                mousePos = (x, y)
                processImage = image.copy()
                cv2.putText(processImage, pointState, mousePos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.circle(processImage, mousePos, circleRadian, color, -1)
                
                if pointState == "Point_TopLeft":
                    return
                
                elif pointState == "Point_BottomLeft":
                    cv2.line(processImage, selectionPoint[0], mousePos, color, 2)
                    return
                
                elif pointState == "Point_BottomRight":
                    cv2.line(processImage, selectionPoint[0], mousePos, color, 2)
                    cv2.line(processImage, selectionPoint[1], mousePos, color, 2)
                    return
                
                elif pointState == "Point_TopRight":
                    cv2.line(processImage, selectionPoint[0], mousePos, color, 2)
                    cv2.line(processImage, selectionPoint[2], mousePos, color, 2)
                    return
            
            if event == cv2.EVENT_LBUTTONDOWN:
                x *= 2
                y *= 2
                
                if pointTL == False:
                    self.perspective_transform_pointTL = [x,y]
                    cv2.putText(image, pointState, (mousePos[0], mousePos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.circle(image, mousePos, circleRadian, color, -1)
                    selectionPoint.append((mousePos))
                    pointTL = True
                    pointState = "Point_BottomLeft"
                    
                elif pointBL == False:
                    self.perspective_transform_pointBL = [x,y]
                    cv2.putText(image, pointState, (mousePos[0], mousePos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.circle(image, mousePos, circleRadian, color, -1)
                    cv2.line(image, selectionPoint[0], mousePos, color, 2)
                    selectionPoint.append((mousePos))
                    pointBL = True
                    pointState = "Point_BottomRight"
                    
                elif pointBR == False:
                    self.perspective_transform_pointBR = [x,y]
                    pointBR = True
                    cv2.putText(image, pointState, (mousePos[0], mousePos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.circle(image, mousePos, circleRadian, color, -1)
                    cv2.line(image, selectionPoint[1], mousePos, color, 2)
                    selectionPoint.append((mousePos))
                    pointState = "Point_TopRight"
                        
                elif pointBT == False:
                    self.perspective_transform_pointBT = [x,y]
                    pointBT = True
                    cv2.putText(image, pointState, (mousePos[0], mousePos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.circle(image, mousePos, circleRadian, color, -1)
                    cv2.line(image, selectionPoint[2], mousePos, color, 2)
                    selectionPoint.append((mousePos))
                    pointState = ""
                
            if pointTL and pointBL and pointBR and pointBT:
                print("Collect Point complete")
                self.isPerspectivePointSet = True        
                cv2.destroyWindow("Select_Transform")
                return
            
        cv2.namedWindow('Select_Transform')
        cv2.setMouseCallback('Select_Transform', selectTransformPoint)
        
        while self.isPerspectivePointSet == False:
            cv2.imshow('Select_Transform', processImage)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or cv2.getWindowProperty('Select_Transform', cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow("Select_Transform")
                break
    
    def selctOCRInspection(self):
        global mousePos, processImage, color, isDrawing, isSelected, mousePosClick, ocr_point_tl, ocr_point_br
    
        image = self.image.copy()
        image = cv2.resize(image, (0,0), fx = 1, fy = 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processImage = image.copy()
        
        ocr_point_tl = (0,0)
        ocr_point_br = (0,0)
        isDrawing = False
        isSelected = False
        mousePos = (0,0)
        mousePosClick = (0,0)
        color = (0, 0, 255)
        
        def selectROIOCR(event, x, y, flags, param):
            global mousePos, processImage, color, isDrawing, isSelected, mousePosClick, ocr_point_tl, ocr_point_br
            if event == cv2.EVENT_MOUSEMOVE:
                mousePos = (x, y)

                if isDrawing:
                    processImage = cv2.rectangle(image.copy(), mousePosClick, mousePos, color, 2)
                    cv2.putText(processImage, "OCR", (mousePosClick[0], mousePosClick[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if event == cv2.EVENT_LBUTTONDOWN:
                processImage = image.copy()
                mousePosClick = mousePos
                isDrawing = True
                
            if event == cv2.EVENT_LBUTTONUP:
                isDrawing = False
                
                if (mousePos[0] < mousePosClick[0] or mousePos[1] < mousePosClick[1]):
                    leftColumn = [mousePos[0], mousePosClick[0]]
                    bottomRow = [mousePos[1], mousePosClick[1]]
                    leftColumn.sort()
                    bottomRow.sort()
                    mousePosClick = (leftColumn[0], bottomRow[0])
                    mousePos = (leftColumn[1], bottomRow[1])
                
                ocr_point_tl = (mousePosClick[0], mousePosClick[1])
                ocr_point_br = (mousePos[0], mousePos[1])
                
                processImage = cv2.rectangle(image.copy(), mousePosClick, mousePos, color, 2)
                cv2.putText(processImage, "OCR", (mousePosClick[0], mousePosClick[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
            if event == cv2.EVENT_RBUTTONDOWN:
                isSelected = True
                cv2.destroyWindow('OCR_Inspection')
                
                if self.debug:
                    img = self.image[ocr_point_tl[1]:ocr_point_br[1], ocr_point_tl[0]:ocr_point_br[0]]
                    cv2.imshow("1", img)
                    cv2.waitKey()
                    cv2.destroyWindow("1")
                
                self.ocrResult = ImageProcessing.OCRInspection(self.image, ocr_point_tl, ocr_point_br)
                return
            
        cv2.namedWindow('OCR_Inspection')
        cv2.setMouseCallback('OCR_Inspection', selectROIOCR)
        
        while isSelected == False:
            cv2.imshow('OCR_Inspection', processImage)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or cv2.getWindowProperty('OCR_Inspection', cv2.WND_PROP_VISIBLE) < 1:
                break
        try:
            cv2.destroyWindow("OCR_Inspection")
        except:
            pass

class ImageProcessing:
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags = cv2.INTER_LINEAR)
        return result
    
    def grayscale(image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return img
    
    def gaussianBlur(image, kernelx, kernely):
        kernelx = round(kernelx)
        kernely = round(kernely)
        if kernelx % 2 != 1:
            kernelx = 1
        if kernely % 2 != 1:
            kernely = 1
        return cv2.GaussianBlur(image,(kernelx,kernely),0)
    
    def rangeMask(image, lower, upper):
        return cv2.inRange(image, lower, upper)
    
    def hsvMask(image, lower, upper):
        hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsvMask = cv2.inRange(hsvImg, np.array(lower), np.array(upper))
        return cv2.bitwise_and(image, image, mask = hsvMask)
    
    def dilation(image, kernelx, kernely):
        kernelDilation = np.ones((kernelx, kernely), np.uint8)
        return cv2.dilate(image, kernelDilation, iterations = 1)
    
    def erosion(image, kernelx, kernely):
        kernelErosion = np.ones((kernelx, kernely), np.uint8)
        return cv2.erode(image, kernelErosion, iterations = 1)
    
    def morphOpen(image, kernelx, kernely):
        morphOpenKernel = np.ones((kernelx, kernely), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, morphOpenKernel)
    
    def morphClose(image, kernelx, kernely):
        morphCloseKernel = np.ones((kernelx, kernely), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, morphCloseKernel)
    
    def morphGradient(image, kernelx, kernely):
        morphGradientKernel = np.ones((kernelx, kernely), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, morphGradientKernel)
    
    def perspectiveTransform(image, pt_TL, pt_BL, pt_BR, pt_TR):
        width_AD = np.sqrt(((pt_TL[0] - pt_TR[0]) ** 2) + ((pt_TL[1] - pt_TR[1]) ** 2))
        width_BC = np.sqrt(((pt_BL[0] - pt_BR[0]) ** 2) + ((pt_BL[1] - pt_BR[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))
        
        height_AB = np.sqrt(((pt_TL[0] - pt_BL[0]) ** 2) + ((pt_TL[1] - pt_BL[1]) ** 2))
        height_CD = np.sqrt(((pt_BR[0] - pt_TR[0]) ** 2) + ((pt_BR[1] - pt_TR[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))
        
        input_pts = np.float32([pt_TL, pt_BL, pt_BR, pt_TR])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])
        
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags = cv2.INTER_LINEAR)
    
    def OCRInspection(image, pt_TL, pt_BR):
        crop_img = image[int(pt_TL[1]):int(pt_BR[1]), int(pt_TL[0]):int(pt_BR[0])]
        
        reader = easyocr.Reader(['en'])
        result = reader.readtext(crop_img)
        
        text = ""
        for detection in result:
            text += detection[1] + " "
        print (text.strip())
        return text.strip()

class Color:
    def __init__(self):
        self.main = "#1F6AA5"
        self.background = "#3A3A3A"
        self.background_frame = "#181717"
        self.white = "#F2F2F2"
        self.red = "#C00000"
        self.green = "#00B050"
        self.blue = "#0078D4"
        self.black = "#0D0D0D"
        self.pink = "#F24171"
        
        self.transparent = "transparent"
        self.disable = "#626262"

class Font:
    def __init__(self):
        self.font = "Kanit light"
        self.font_bold = "Kanit" 

class MainApp(CTk):
    def __init__(self):
        super().__init__()

        # Variable
        self.isGray = False
        self.isBlur = False
        self.isMask_hsv = False
        self.isErosion = False
        self.isDilation = False
        self.isMorphOpen = False
        self.isMorphClose = False
        
        self.comboBox_var_blur = ["Gaussian Blurring", "Averaging", "Median Blurring", "Bilateral Filtering "]
        
        # Screen Setting
        screen_width = 1600
        screen_height = 900
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        self.resizable(False, False)
        
        set_appearance_mode("Dark")
        self.title("Custom OpenCV")
        self.bind("<Escape>", lambda event : self.exitApplication())
        self.protocol("WM_DELETE_WINDOW", self.exitApplication)
        
        self.camera = ThreadCamera()
        self.camera.start()
        
        self.build_ui()
        self.FixedUpdate()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure((0,2), weight = 0)
        self.grid_columnconfigure(1, weight = 0, minsize = 5)
        
        """ Left """
        self.frame_camera = CTkFrame(self, fg_color = Color().transparent)
        self.frame_camera.grid(row = 0, column = 0)
        
        self.frame_camera_selection = CTkFrame(self.frame_camera, fg_color = Color().transparent)
        # self.frame_camera_selection.pack()
        
        self.label_camera_number_title = CTkLabel(self.frame_camera_selection, text = "Camera Number : ")
        self.label_camera_number_title.grid(row = 0, column = 0)
        
        self.comboBox_camera_number = CTkComboBox(self.frame_camera_selection, values = self.camera.cameraList, 
                                                width = 70,
                                                command = self.combobox_camera_number_callback)
        self.comboBox_camera_number.grid(row = 0, column = 1)
        
        self.displayCamera = CTkEntry(self.frame_camera, fg_color = Color().black,
                                      width = int(2592/2.5),
                                      height = int(1944/2.5),
                                      corner_radius = 0,
                                      justify = CENTER)
        self.displayCamera.insert(0, "Camera Display")
        self.displayCamera.configure(state = "disabled")
        self.displayCamera.pack()
        
        self.showCameraImage_ui = CTkLabel(self.displayCamera, text = "",
                                    fg_color = Color().transparent)
        self.showCameraImage_ui.place(relx = 0.5, rely = 0.5, anchor = CENTER)
        
        self.slider_rotate = CTkSlider(self.frame_camera,
                                    width = int(2592/2.5),
                                    height = 20,
                                    progress_color = Color().blue,
                                    fg_color = Color().blue,
                                    button_color = Color().white,
                                    button_hover_color = Color().white,
                                    from_ = -180, to = 180,
                                    number_of_steps = 720,
                                    command = self.slider_rotate_callback)
        self.slider_rotate.pack()
        self.slider_rotate.bind("<Double-Button-1>", self.slider_rotate_doubleClick_callback)
        
        self.label_rotate = CTkLabel(self.frame_camera, text = "0°")
        self.label_rotate.pack()
        
        self.btn_cameraSnapshot = CTkButton(self, text = "Camera Snapshot", command = self.camera.snapshotImage)
        self.btn_cameraSnapshot.place(relx = 0.9, rely = 0, anchor = NE)
        
        self.btn_processImageSnapshot = CTkButton(self, text = "Process Image Snapshot", command = self.camera.snapshotProcessImage)
        self.btn_processImageSnapshot.place(relx = 1, rely = 0, anchor = NE)
        
        self.frame_OCR = CTkFrame(self.frame_camera, fg_color = Color().transparent)
        self.frame_OCR.pack()
        self.frame_OCR.grid_rowconfigure(0, weight = 0)
        self.frame_OCR.grid_columnconfigure(0, weight = 0)
        
        self.label_OCR_title = CTkLabel(self.frame_OCR, text = "OCR Result :",
                                        font = (Font().font_bold, 20))
        self.label_OCR_title.grid(row = 0, column = 0, padx = (0,10))
        
        self.entry_ocrInspection_result = CTkEntry(self.frame_OCR, 
                                                   width = 800, height = 30,
                                                   font = (Font().font, 20))
        self.entry_ocrInspection_result.grid(row = 0, column = 1)
        
        """ Right """
        self.mainframe = CTkFrame(self, fg_color = Color().transparent)
        self.mainframe.grid(row = 0, column = 2)
        
        self.checkbox_isPerspectiveTransform = CTkCheckBox(self.mainframe, text = "Perspective Transform", command = self.checkbox_perspectiveTransform_callback)
        self.checkbox_isPerspectiveTransform.pack()
        self.btn_savePerspectiveTransform = CTkButton(self.mainframe, text = "Reset Perspective Transform", command = self.btn_resetPerspectiveTransform_callback)
        self.btn_savePerspectiveTransform.pack()
        
        self.checkbox_isGray = CTkCheckBox(self.mainframe, text = "Grayscale", command = self.checkbox_gray_callback)
        self.checkbox_isGray.pack()
        
        self.checkbox_isMask_hsv = CTkCheckBox(self.mainframe, text = "HSV Mask", command = self.checkbox_mask_hsv_callback)
        # self.checkbox_isMask_hsv.pack()
        
        self.slider_mask_hsv = Slider_MaskHSV(self.mainframe, self)
        # self.slider_mask_hsv.pack()
        
        self.module_blur_gussian = Module_Blur_Gaussian(self.mainframe, self)
        self.module_blur_gussian.set_callback(self.module_blur_gussian_callback)
        self.module_blur_gussian.pack()
        
        self.module_mask_gray = Module_Mask_Gray(self.mainframe, self)
        self.module_mask_gray.set_callback(self.module_mask_gray_callback)
        self.module_mask_gray.pack()
        
        self.module_erosion = Module_Erosion(self.mainframe, self)
        self.module_erosion.set_callback(self.module_erosion_callback)
        self.module_erosion.pack()
        
        self.module_dilation = Module_Dilation(self.mainframe, self)
        self.module_dilation.set_callback(self.module_dilation_callback)
        self.module_dilation.pack()
        
        self.module_morph_open = Module_MorphOpen(self.mainframe, self)
        self.module_morph_open.set_callback(self.module_morph_open_callback)
        self.module_morph_open.pack()
        
        self.module_morph_close = Module_MorphClose(self.mainframe, self)
        self.module_morph_close.set_callback(self.module_morph_close_callback)
        self.module_morph_close.pack()
        
        self.module_morph_gradient = Module_MorphGradient(self.mainframe, self)
        self.module_morph_gradient.set_callback(self.module_morph_gradient_callback)
        self.module_morph_gradient.pack()
        
        self.btn_OCRInspection = CTkButton(self.mainframe, text = "OCR Inspection", command = self.btn_OCRInspection_callback)
        self.btn_OCRInspection.pack()

    def FixedUpdate(self):
        self.cameraUpdate()
        self.after(10, self.FixedUpdate)

    def combobox_camera_number_callback(self, choice):
        self.camera.changeCameraNumber(int(choice))

    def slider_rotate_callback(self, value):
        self.camera.rotateAngle = value
        self.label_rotate.configure(text = f"{self.camera.rotateAngle}°")
        
    def slider_rotate_doubleClick_callback(self, event):
        self.slider_rotate.set(0)
        self.camera.rotateAngle = 0
        self.label_rotate.configure(text = f"{self.camera.rotateAngle}")
    
    def checkbox_gray_callback(self):
        if self.checkbox_isGray.get():
            self.camera.isGray = True
            self.camera.add_imageProcessing(self.camera.imageProcessing_grayscale)
        else:
            self.camera.isGray = False
            self.camera.remove_imageProcessing(self.camera.imageProcessing_grayscale)
    
    def checkbox_mask_hsv_callback(self):
        if self.checkbox_isMask_hsv.get():
            self.camera.isMask_hsv = True
        else:
            self.camera.isMask_hsv = False
    
    def slider_mask_hsv_callback(self):
        self.camera.mask_hsv_upper = self.slider_mask_hsv.upper_range
        self.camera.mask_hsv_lower = self.slider_mask_hsv.lower_range
    
    def module_blur_gussian_callback(self):
        self.camera.isBlur_gussian = self.module_blur_gussian.isActive
        self.camera.blur_gussian_kernel_x = self.module_blur_gussian.kernelx
        self.camera.blur_gussian_kernel_y = self.module_blur_gussian.kernely
        
        if self.module_blur_gussian.isActive:
            self.camera.add_imageProcessing(self.camera.imageProcessing_blur_gaussian)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_blur_gaussian)
    
    def module_mask_gray_callback(self):
        self.camera.isMask_gray = self.module_mask_gray.isActive
        self.camera.mask_gray_lower = self.module_mask_gray.mask_lower
        self.camera.mask_gray_upper = self.module_mask_gray.mask_upper
        
        if self.module_mask_gray.isActive:
            self.camera.add_imageProcessing(self.camera.imageProcessing_mask_gray)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_mask_gray)

    def module_erosion_callback(self):
        self.camera.isErosion = self.module_erosion.isActive
        self.camera.erosion_kernel_x = self.module_erosion.kernelx
        self.camera.erosion_kernel_y = self.module_erosion.kernely
        
        if self.module_erosion.isActive:
            self.camera.add_imageProcessing(self.camera.imageProcessing_erosion)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_erosion)
    
    def module_dilation_callback(self):
        self.camera.isDilation = self.module_dilation.isActive
        self.camera.dilation_kernel_x = self.module_dilation.kernelx
        self.camera.dilation_kernel_y = self.module_dilation.kernely
        
        if self.module_dilation.isActive:
            self.camera.add_imageProcessing(self.camera.imageProcessing_dilation)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_dilation)
        
    def module_morph_open_callback(self):
        self.camera.isMorphOpen = self.module_morph_open.isActive
        self.camera.morph_open_kernel_x = self.module_morph_open.kernelx
        self.camera.morph_open_kernel_y = self.module_morph_open.kernely
        
        if self.module_morph_open.isActive:
            self.camera.add_imageProcessing(self.camera.imageProcessing_morph_open)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_morph_open)
        
    def module_morph_close_callback(self):
        self.camera.isMorphClose = self.module_morph_close.isActive
        self.camera.morph_close_kernel_x = self.module_morph_close.kernelx
        self.camera.morph_close_kernel_y = self.module_morph_close.kernely
    
        if self.module_morph_close.isActive:
            self.camera.add_imageProcessing(self.camera.imageProcessing_morph_close)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_morph_close)
    
    def module_morph_gradient_callback(self):
        self.camera.isMorphGradient = self.module_morph_gradient.isActive
        self.camera.morph_gradient_kernel_x = self.module_morph_gradient.kernelx
        self.camera.morph_gradient_kernel_y = self.module_morph_gradient.kernely
    
        if self.module_morph_gradient.isActive:
            self.camera.add_imageProcessing(self.camera.imageProcessing_morph_gradient)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_morph_gradient)
        return
    
    def cameraUpdate(self):
        try:
            self.processImage = Image.fromarray(self.camera.showImage)
            self.showCameraImage_ui.configure(image = CTkImage(self.processImage, size = (int(self.camera.width/2.5), self.camera.height/2.5)))
            
            # Update if cancel perspective select point
            if self.camera.isPerspectiveTransform:    
                self.checkbox_isPerspectiveTransform.select()
            else:
                self.checkbox_isPerspectiveTransform.deselect()
        except:
            pass
    
    def checkbox_perspectiveTransform_callback(self):
        if self.checkbox_isPerspectiveTransform.get():
            self.camera.isPerspectiveTransform = True
        else:
            self.camera.isPerspectiveTransform = False
    
    def btn_resetPerspectiveTransform_callback(self):
        self.camera.isPerspectiveTransform = False
        self.camera.isPerspectivePointSet = False
        
    def btn_OCRInspection_callback(self):
        self.camera.selctOCRInspection()
        self.entry_ocrInspection_result.delete(0, END)
        self.entry_ocrInspection_result.insert(1, self.camera.ocrResult)
    
    def exitApplication(self):
        self.camera.videoCapture.release()
        self.camera.stop()
        self.destroy()

class Slider_MaskHSV(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.configure(fg_color = Color().transparent)

        self.mainApp = mainApp

        self.upper_range = [255,255,255]
        self.lower_range = [0,0,0]
    
        self.build_ui()

    def build_ui(self):
        self.grid_rowconfigure((0,2), weight = 0)
        self.grid_rowconfigure(1, weight = 0, minsize = 5)
        
        self.slider_mask_hsv_lower = HSVSlider(self, "Lower")
        self.slider_mask_hsv_lower.grid(row = 0, column = 0)
        
        self.slider_mask_hsv_upper = HSVSlider(self, "Upper")
        self.slider_mask_hsv_upper.grid(row = 2, column = 0)
        self.slider_mask_hsv_upper.slider_mask_hsv_h.set(0)
        self.slider_mask_hsv_upper.slider_mask_hsv_s.set(0)
        self.slider_mask_hsv_upper.slider_mask_hsv_v.set(1)
        self.slider_mask_hsv_upper.label_mask_hsv_v_value.configure(text = "1.000")
        
    def updateValue(self):
        self.upper_range = list(self.slider_mask_hsv_upper.rgb)
        self.lower_range = list(self.slider_mask_hsv_lower.rgb)
        
        self.mainApp.slider_mask_hsv_callback()

class Module_Blur_Gaussian(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.kernelx = 1
        self.kernely = 1
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 0, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure((0,2), weight = 0)
        self.frame_border.grid_rowconfigure((1), weight = 0, minsize = 10)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_active = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_active.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_active.grid_columnconfigure((0,2), weight = 0)
        self.frame_active.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_active.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_active = CTkCheckBox(self.frame_active, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_active_callback)
        self.checkbox_active.grid(row = 0, column = 0)
        
        self.label_active_title = CTkLabel(self.frame_active, text = "Gussian Blur",
                                      height = 15)
        self.label_active_title.grid(row = 0, column = 2)
        
        self.slider_kernel = Slider_Kernel_xy(self.frame_border)
        self.slider_kernel.set_callback(self.updateValue)
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 255, number_of_steps = 127)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 255, number_of_steps = 127)
        self.slider_kernel.grid(row = 2, column = 0)
    
    def checkbox_active_callback(self):
        if self.checkbox_active.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def updateValue(self):
        self.kernelx = self.slider_kernel.kernelx
        self.kernely = self.slider_kernel.kernely
        
        if self.callback:
            self.callback()

class Module_Mask_Gray(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.mask_lower = 0
        self.mask_upper = 255
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 0, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure((0,2), weight = 0)
        self.frame_border.grid_rowconfigure((1), weight = 0, minsize = 10)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_active = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_active.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_active.grid_columnconfigure((0,2), weight = 0)
        self.frame_active.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_active.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_active = CTkCheckBox(self.frame_active, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_active_callback)
        self.checkbox_active.grid(row = 0, column = 0)
        
        self.label_active_title = CTkLabel(self.frame_active, text = "Grayscale Mask",
                                      height = 15)
        self.label_active_title.grid(row = 0, column = 2)
        
        self.slider = Slider_xy(self.frame_border)
        self.slider.set_callback(self.updateValue)
        self.slider.slider_x.set(0)
        self.slider.slider_y.set(255)
        self.slider.label_x_value.configure(text = "0")
        self.slider.label_x_title.configure(text = "Lower")
        self.slider.label_y_value.configure(text = "255")
        self.slider.label_y_title.configure(text = "Upper")
        self.slider.grid(row = 2, column = 0)
    
    def checkbox_active_callback(self):
        if self.checkbox_active.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def updateValue(self):
        self.mask_lower = self.slider.x
        self.mask_upper = self.slider.y
        
        if self.callback:
            self.callback()

class Module_Erosion(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.kernelx = 1
        self.kernely = 1
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 0, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure((0,2), weight = 0)
        self.frame_border.grid_rowconfigure((1), weight = 0, minsize = 10)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_active = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_active.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_active.grid_columnconfigure((0,2), weight = 0)
        self.frame_active.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_active.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_active = CTkCheckBox(self.frame_active, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_active_callback)
        self.checkbox_active.grid(row = 0, column = 0)
        
        self.label_active_title = CTkLabel(self.frame_active, text = "Erosion",
                                      height = 15)
        self.label_active_title.grid(row = 0, column = 2)
        
        self.slider_kernel = Slider_Kernel_xy(self.frame_border)
        self.slider_kernel.set_callback(self.updateValue)
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.grid(row = 2, column = 0)
    
    def checkbox_active_callback(self):
        if self.checkbox_active.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def updateValue(self):
        self.kernelx = self.slider_kernel.kernelx
        self.kernely = self.slider_kernel.kernely
        
        if self.callback:
            self.callback()

class Module_Dilation(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.kernelx = 1
        self.kernely = 1
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 0, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure((0,2), weight = 0)
        self.frame_border.grid_rowconfigure((1), weight = 0, minsize = 10)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_active = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_active.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_active.grid_columnconfigure((0,2), weight = 0)
        self.frame_active.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_active.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_active = CTkCheckBox(self.frame_active, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_active_callback)
        self.checkbox_active.grid(row = 0, column = 0)
        
        self.label_active_title = CTkLabel(self.frame_active, text = "Dilation",
                                      height = 15)
        self.label_active_title.grid(row = 0, column = 2)
        
        self.slider_kernel = Slider_Kernel_xy(self.frame_border)
        self.slider_kernel.set_callback(self.updateValue)
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.grid(row = 2, column = 0)
    
    def checkbox_active_callback(self):
        if self.checkbox_active.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def updateValue(self):
        self.kernelx = self.slider_kernel.kernelx
        self.kernely = self.slider_kernel.kernely
        
        if self.callback:
            self.callback()

class Module_MorphOpen(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.kernelx = 1
        self.kernely = 1
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 0, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure((0,2), weight = 0)
        self.frame_border.grid_rowconfigure((1), weight = 0, minsize = 10)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_active = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_active.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_active.grid_columnconfigure((0,2), weight = 0)
        self.frame_active.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_active.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_active = CTkCheckBox(self.frame_active, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_active_callback)
        self.checkbox_active.grid(row = 0, column = 0)
        
        self.label_active_title = CTkLabel(self.frame_active, text = "Morph Open",
                                      height = 15)
        self.label_active_title.grid(row = 0, column = 2)
        
        self.slider_kernel = Slider_Kernel_xy(self.frame_border)
        self.slider_kernel.set_callback(self.updateValue)
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.grid(row = 2, column = 0)
    
    def checkbox_active_callback(self):
        if self.checkbox_active.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def updateValue(self):
        self.kernelx = self.slider_kernel.kernelx
        self.kernely = self.slider_kernel.kernely
        
        if self.callback:
            self.callback()

class Module_MorphClose(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.kernelx = 1
        self.kernely = 1
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 0, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure((0,2), weight = 0)
        self.frame_border.grid_rowconfigure((1), weight = 0, minsize = 10)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_active = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_active.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_active.grid_columnconfigure((0,2), weight = 0)
        self.frame_active.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_active.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_active = CTkCheckBox(self.frame_active, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_active_callback)
        self.checkbox_active.grid(row = 0, column = 0)
        
        self.label_active_title = CTkLabel(self.frame_active, text = "Morph Close",
                                      height = 15)
        self.label_active_title.grid(row = 0, column = 2)
        
        self.slider_kernel = Slider_Kernel_xy(self.frame_border)
        self.slider_kernel.set_callback(self.updateValue)
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 10, number_of_steps = 10)
        self.slider_kernel.grid(row = 2, column = 0)
    
    def checkbox_active_callback(self):
        if self.checkbox_active.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def updateValue(self):
        self.kernelx = self.slider_kernel.kernelx
        self.kernely = self.slider_kernel.kernely
        
        if self.callback:
            self.callback()

class Module_MorphGradient(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.kernelx = 1
        self.kernely = 1
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 0, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure((0,2), weight = 0)
        self.frame_border.grid_rowconfigure((1), weight = 0, minsize = 10)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_active = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_active.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_active.grid_columnconfigure((0,2), weight = 0)
        self.frame_active.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_active.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_active = CTkCheckBox(self.frame_active, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_active_callback)
        self.checkbox_active.grid(row = 0, column = 0)
        
        self.label_active_title = CTkLabel(self.frame_active, text = "Morph Gradient",
                                      height = 15)
        self.label_active_title.grid(row = 0, column = 2)
        
        self.slider_kernel = Slider_Kernel_xy(self.frame_border)
        self.slider_kernel.set_callback(self.updateValue)
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 20, number_of_steps = 20)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 20, number_of_steps = 20)
        self.slider_kernel.grid(row = 2, column = 0)
    
    def checkbox_active_callback(self):
        if self.checkbox_active.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def updateValue(self):
        self.kernelx = self.slider_kernel.kernelx
        self.kernely = self.slider_kernel.kernely
        
        if self.callback:
            self.callback()

class HSVSlider(CTkFrame):
    def __init__(self, master, title:str, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.configure(fg_color = Color().transparent)
        self.grid_rowconfigure(0, weight = 0)
        self.grid_columnconfigure((0,1), weight = 0)
        
        self.title = title
        
        self.hsv = (0,0,0)
        self.rgb = (0,0,0)
        self.hex = "#000000"
        
        self.h = 0
        self.s = 0
        self.v = 0
        
        self.frame_title = CTkFrame(self, fg_color = Color().transparent)
        self.frame_title.grid(row = 0, column = 0, padx = (0,5))
        
        self.label_mask_hsv_title = CTkLabel(self.frame_title, text = self.title,
                                             fg_color = Color().transparent)
        self.label_mask_hsv_title.grid(row = 0, column = 0)
        
        self.box_color = CTkLabel(self.frame_title, text = "",
                                  fg_color = Color().black,
                                  height = 10)
        self.box_color.grid(row = 1, column = 0, sticky = EW)
        
        self.frame_mask_hsv = CTkFrame(self, fg_color = Color().transparent)
        self.frame_mask_hsv.grid(row = 0, column = 1)
        self.frame_mask_hsv.grid_columnconfigure((0,2,4), weight = 0)
        self.frame_mask_hsv.grid_columnconfigure((1,3), weight = 0, minsize = 5)
        self.frame_mask_hsv.grid_rowconfigure((0,1,2), weight = 1)
        
        self.label_mask_hsv_h_title = CTkLabel(self.frame_mask_hsv, text = "H", height = 15)
        self.label_mask_hsv_h_title.grid(row = 0, column = 0)
        
        self.slider_mask_hsv_h = CTkSlider(self.frame_mask_hsv,
                                    height = 15,
                                    progress_color = Color().blue,
                                    fg_color = Color().blue,
                                    button_color = Color().white,
                                    button_hover_color = Color().white,
                                    from_ = 0, to = 359,
                                    number_of_steps = 360,
                                    corner_radius = 0,
                                    button_length = 1,
                                    button_corner_radius = 3,
                                    command = self.slider_h_callback)
        self.slider_mask_hsv_h.set(0)
        self.slider_mask_hsv_h.grid(row = 0, column = 2)
        
        self.label_mask_hsv_h_value = CTkLabel(self.frame_mask_hsv, text = "0", height = 15)
        self.label_mask_hsv_h_value.grid(row = 0, column = 4)
        
        self.label_mask_hsv_s_title = CTkLabel(self.frame_mask_hsv, text = "S", height = 15)
        self.label_mask_hsv_s_title.grid(row = 1, column = 0)
        
        self.slider_mask_hsv_s = CTkSlider(self.frame_mask_hsv,
                                    height = 15,    
                                    progress_color = Color().blue,
                                    fg_color = Color().blue,
                                    button_color = Color().white,
                                    button_hover_color = Color().white,
                                    from_ = 0, to = 1,
                                    number_of_steps = 1000,
                                    corner_radius = 0,
                                    button_length = 1,
                                    button_corner_radius = 3,
                                    command = self.slider_s_callback)
        self.slider_mask_hsv_s.set(0)
        self.slider_mask_hsv_s.grid(row = 1, column = 2)
        
        self.label_mask_hsv_s_value = CTkLabel(self.frame_mask_hsv, text = "0.000", height = 15)
        self.label_mask_hsv_s_value.grid(row = 1, column = 4)
        
        self.label_mask_hsv_v_title = CTkLabel(self.frame_mask_hsv, text = "V", height = 15)
        self.label_mask_hsv_v_title.grid(row = 2, column = 0)
        
        self.slider_mask_hsv_v = CTkSlider(self.frame_mask_hsv,
                                    height = 15,
                                    progress_color = Color().blue,
                                    fg_color = Color().blue,
                                    button_color = Color().white,
                                    button_hover_color = Color().white,
                                    from_ = 0, to = 1,
                                    number_of_steps = 1000,
                                    corner_radius = 0,
                                    button_length = 1,
                                    button_corner_radius = 3,
                                    command = self.slider_v_callback)
        self.slider_mask_hsv_v.set(0)
        self.slider_mask_hsv_v.grid(row = 2, column = 2)
        
        self.label_mask_hsv_v_value = CTkLabel(self.frame_mask_hsv, text = "0.000", height = 15)
        self.label_mask_hsv_v_value.grid(row = 2, column = 4)
        
    def slider_h_callback(self, value):
        self.h = int(value)
        self.label_mask_hsv_h_value.configure(text = self.h)
        self.changeColor()
        self.changeLabelColor()
    
    def slider_s_callback(self,value):
        self.s = round(value,3)
        self.label_mask_hsv_s_value.configure(text = "{:.3f}".format(self.s))
        self.changeColor()
        self.changeLabelColor()
        
    def slider_v_callback(self,value):
        self.v = round(value,3)
        self.label_mask_hsv_v_value.configure(text = "{:.3f}".format(self.v))  
        self.changeColor()
        self.changeLabelColor()
    
    def changeColor(self):
        rgb = colorsys.hsv_to_rgb((self.h/360), self.s, self.v)
        rgb = tuple([int(255 * x) for x in rgb])
        self.rgb = rgb
        self.hex = '#%02x%02x%02x' % self.rgb
        
        self.master.updateValue()
        
    def changeLabelColor(self):
        self.box_color.configure(fg_color = self.hex)

class Slider_Kernel_xy(CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.configure(fg_color = Color().transparent)

        self.isActive = False
        self.kernelx = 1
        self.kernely = 1
        self.callback = None
    
        self.build_ui()
    
    def build_ui(self):
        self.frame_main = CTkFrame(self, fg_color = Color().transparent)
        self.frame_main.grid(row = 0, column = 0) 
        self.frame_main.grid_columnconfigure((0,1,2), weight = 1)
        self.frame_main.grid_rowconfigure(0, weight = 1)
        
        self.frame_title = CTkFrame(self.frame_main, fg_color = Color().transparent)
        self.frame_title.grid(row = 0, column = 0, padx = (0,5))
        
        self.label_kernelx_title = CTkLabel(self.frame_title, text = "Kernel X",
                                      height = 15)
        self.label_kernelx_title.grid(row = 0, column = 0)
        
        self.label_kernely_title = CTkLabel(self.frame_title, text = "Kernel Y",
                                      height = 15)
        self.label_kernely_title.grid(row = 1, column = 0)
        
        self.frame_slider = CTkFrame(self.frame_main, fg_color = Color().transparent)
        self.frame_slider.grid(row = 0, column = 1)
        
        self.slider_kernelx = CTkSlider(self.frame_slider,
                                            height = 15,
                                            progress_color = Color().blue,
                                            fg_color = Color().blue,
                                            button_color = Color().white,
                                            button_hover_color = Color().white,
                                            from_ = 1, to = 255,
                                            number_of_steps = 255,
                                            corner_radius = 0,
                                            button_length = 1,
                                            button_corner_radius = 3,
                                            command = self.slider_kernelx_callback)
        self.slider_kernelx.set(1)
        self.slider_kernelx.grid(row = 0, column = 0)
        
        self.slider_kernely = CTkSlider(self.frame_slider,
                                            height = 15,
                                            progress_color = Color().blue,
                                            fg_color = Color().blue,
                                            button_color = Color().white,
                                            button_hover_color = Color().white,
                                            from_ = 1, to = 255,
                                            number_of_steps = 255,
                                            corner_radius = 0,
                                            button_length = 1,
                                            button_corner_radius = 3,
                                            command = self.slider_kernely_callback)
        self.slider_kernely.set(1)
        self.slider_kernely.grid(row = 1, column = 0)
        
        self.frame_value = CTkFrame(self.frame_main, fg_color = Color().transparent)
        self.frame_value.grid(row = 0, column = 2)
        
        self.label_kernelx_value = CTkLabel(self.frame_value, text = "1",
                                           height = 15)
        self.label_kernelx_value.grid(row = 0, column = 0)
        
        self.label_kernely_value = CTkLabel(self.frame_value, text = "1",
                                           height = 15)
        self.label_kernely_value.grid(row = 1, column = 0)
        
    def slider_kernelx_callback(self, value):
        value = int(value)
        self.kernelx = value
        self.label_kernelx_value.configure(text = str(value))
        
        if self.callback:
            self.callback()
    
    def slider_kernely_callback(self, value):
        value = int(value)
        self.kernely = value
        self.label_kernely_value.configure(text = str(value))
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback

class Slider_xy(CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.configure(fg_color = Color().transparent)

        self.isActive = False
        self.x = 1
        self.y = 1
        self.callback = None
    
        self.build_ui()
    
    def build_ui(self):
        self.frame_main = CTkFrame(self, fg_color = Color().transparent)
        self.frame_main.grid(row = 0, column = 0) 
        self.frame_main.grid_columnconfigure((0,1,2), weight = 1)
        self.frame_main.grid_rowconfigure(0, weight = 1)
        
        self.frame_title = CTkFrame(self.frame_main, fg_color = Color().transparent)
        self.frame_title.grid(row = 0, column = 0, padx = (0,5))
        
        self.label_x_title = CTkLabel(self.frame_title, text = "X",
                                      height = 15)
        self.label_x_title.grid(row = 0, column = 0)
        
        self.label_y_title = CTkLabel(self.frame_title, text = "Y",
                                      height = 15)
        self.label_y_title.grid(row = 1, column = 0)
        
        self.frame_slider = CTkFrame(self.frame_main, fg_color = Color().transparent)
        self.frame_slider.grid(row = 0, column = 1)
        
        self.slider_x = CTkSlider(self.frame_slider,
                                            height = 15,
                                            progress_color = Color().blue,
                                            fg_color = Color().blue,
                                            button_color = Color().white,
                                            button_hover_color = Color().white,
                                            from_ = 0, to = 255,
                                            number_of_steps = 256,
                                            corner_radius = 0,
                                            button_length = 1,
                                            button_corner_radius = 3,
                                            command = self.slider_x_callback)
        self.slider_x.set(1)
        self.slider_x.grid(row = 0, column = 0)
        
        self.slider_y = CTkSlider(self.frame_slider,
                                            height = 15,
                                            progress_color = Color().blue,
                                            fg_color = Color().blue,
                                            button_color = Color().white,
                                            button_hover_color = Color().white,
                                            from_ = 0, to = 255,
                                            number_of_steps = 256,
                                            corner_radius = 0,
                                            button_length = 1,
                                            button_corner_radius = 3,
                                            command = self.slider_y_callback)
        self.slider_y.set(1)
        self.slider_y.grid(row = 1, column = 0)
        
        self.frame_value = CTkFrame(self.frame_main, fg_color = Color().transparent)
        self.frame_value.grid(row = 0, column = 2)
        
        self.label_x_value = CTkLabel(self.frame_value, text = "1",
                                           height = 15)
        self.label_x_value.grid(row = 0, column = 0)
        
        self.label_y_value = CTkLabel(self.frame_value, text = "1",
                                           height = 15)
        self.label_y_value.grid(row = 1, column = 0)
        
    def slider_x_callback(self, value):
        value = int(value)
        self.x = value
        self.label_x_value.configure(text = str(value))
        
        if self.callback:
            self.callback()
    
    def slider_y_callback(self, value):
        value = int(value)
        self.y = value
        self.label_y_value.configure(text = str(value))
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
 
app = MainApp()
app.mainloop()