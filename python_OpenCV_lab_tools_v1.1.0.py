from customtkinter import *
import cv2
from PIL import Image
import numpy as np
import pywinstyles
import threading
import colorsys
import time
from datetime import datetime
import math
import easyocr

class ThreadCamera(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.debug = False
        self.daemon = True
        
        self.isSelectCamera = False
        self.cameraList = []
        self.cameraNumber = 0
        self.width = 2592
        self.height = 1944
        self.ratio = (4, 3)
        self.scaledown = 40/100 # percentage
        self.videoCapture = None
        
        self.fps = 0
        self.start_time = time.time()
        self.frameCount = 0
        
        self.image = None
        self.showImage = None
        
        self.isPerspectiveTransform = False
        self.isPerspectivePointSet = False
        
        self.isInvert = False
        self.isGray = False
        self.isGrayChannel = False
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
        
        self.gray_channel = 0
        
        self.mask_red = False
        self.mask_green = False
        self.mask_blue = False
        
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
        
    def run(self):
        while self.isSelectCamera == False:
            time.sleep(0.1)
            pass
        
        self.cameraSetup(self.cameraNumber)
        
        while not self._stop_event.is_set():
            try:
                self.ret, self.frame = self.videoCapture.read()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.calculateFPS()
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
            
                self.showImage = cv2.resize(image,(0,0),fx = self.scaledown, fy = self.scaledown)
                self.image = image
                
            except Exception as error:
                try:
                    image = self.showErrorImage()
                    
                    self.showImage = cv2.resize(image,(0,0),fx = self.scaledown, fy = self.scaledown)
                    self.image = image
                except:
                    pass
        print("Camera thread stop")
    
    def stop(self):
        self._stop_event.set()
    
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
        
        # Print Order Function List
        print(self.imageProcessingOrderName)
    
    def reset_imageProcessing(self):
        self.imageProcessingOrder.clear()
        self.imageProcessingOrderName.clear()
    
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
    
    def imageProcessing_invert(self, image):
        return ImageProcessing.invert(image)
    
    def imageProcessing_mask_red(self, image):
        return ImageProcessing.removeRed_channel(image)
    
    def imageProcessing_mask_green(self, image):
        return ImageProcessing.removeGreen_channel(image)
    
    def imageProcessing_mask_blue(self, image):
        return ImageProcessing.removeBlue_channel(image)
    
    def imageProcessing_grayscale(self, image):
        return ImageProcessing.grayscale(image)
    
    def imageProcessing_grayscale_channel(self, image):
        return ImageProcessing.grayscale_channel(image, self.gray_channel)
    
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
    
    def cameraSetup(self, cameraNumber : int):
        self.videoCapture = cv2.VideoCapture(cameraNumber)
        
        greatest_common_divisor = math.gcd(self.width, self.height)
        w_ratio = self.width // greatest_common_divisor
        h_ratio = self.height // greatest_common_divisor
        self.ratio = (w_ratio, h_ratio)
        
        self.videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    
    def changeCameraNumber(self, cameraNumber):
        if cameraNumber != self.cameraNumber:
            self.cameraNumber = cameraNumber
            self.videoCapture.release()
            self.videoCapture = cv2.VideoCapture(self.cameraNumber)
            self.videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    
    def showErrorImage(self):
        image = self.frame
        
        text = "IMAGE ERROR"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = (image.shape[1] - text_width) // 2
        text_y = (image.shape[0] + text_height) // 2
        
        rect_width = text_width + 100
        rect_height = text_height + 50
        rect_color = (0, 0, 0)
        rect_x = (image.shape[1] - rect_width) // 2
        rect_y = (image.shape[0] - rect_height) // 2
        
        cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rect_color, -1)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255,255,255), font_thickness)
        return image
    
    def calculateFPS(self):
        current_time = time.time()
        self.frameCount += 1
        
        elapsed_time = current_time - self.start_time
        if elapsed_time > 1:
            fps = self.frameCount / elapsed_time
            self.fps = f"{fps:.2f}"
            self.frameCount = 0
            self.start_time = current_time
    
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
            
        cv2.namedWindow('Select_Transform', cv2.WINDOW_NORMAL)
        if self.ratio == (4, 3):
            cv2.resizeWindow('Select_Transform', 1200,900)
        else:
            cv2.resizeWindow('Select_Transform', 1600,900)
        cv2.setMouseCallback('Select_Transform', selectTransformPoint)
        
        while self.isPerspectivePointSet == False:
            cv2.imshow('Select_Transform', processImage)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or cv2.getWindowProperty('Select_Transform', cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow("Select_Transform")
                break
    
    def selectOCRInspection(self):
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
                cv2.rectangle(processImage, (mousePosClick[0], mousePos[1]), (mousePosClick[0] + 350, mousePos[1] + 50), color, -1)
                cv2.putText(processImage, "R-Click : Start OCR", (mousePosClick[0] + 10, mousePos[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
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
            
        cv2.namedWindow('OCR_Inspection', cv2.WINDOW_NORMAL)
        if self.ratio == (4, 3):
            cv2.resizeWindow('OCR_Inspection', 1200,900)
        else:
            cv2.resizeWindow('OCR_Inspection', 1600,900)
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
    
    def invert(image):
        img  = cv2.bitwise_not(image)
        return img
    
    def grayscale(image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return img
    
    def grayscale_channel(image, channel):
        img = image[:,:,channel]
        return img
    
    def removeRed_channel(image):
        image[:,:,0] = np.zeros([image.shape[0], image.shape[1]])
        return image
    
    def removeGreen_channel(image):
        image[:,:,1] = np.zeros([image.shape[0], image.shape[1]])
        return image
    
    def removeBlue_channel(image):
        image[:,:,2] = np.zeros([image.shape[0], image.shape[1]])
        return image
    
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
        
        self.module_label = (self.font_bold, 14)

class MainApp(CTk):
    def __init__(self):
        super().__init__()

        # Variable
        self.comboBox_var_blur = ["Gaussian Blurring", "Averaging", "Median Blurring", "Bilateral Filtering "]
        
        self.animation_silder_label_isplay = False
        self.animation_slider_label_timer = 0
        
        # Screen Setting
        screen_width = 1600
        screen_height = 900
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        self.resizable(False, False)
        
        set_appearance_mode("Dark")
        
        self.title("OpenCV Lab Tools")
        
        self.protocol("WM_DELETE_WINDOW", self.exitApplication)
        self.bind("<Escape>", lambda event : self.exitApplication())
        self.bind("<F1>", self.program_keyboard_command)
        
        self.toplevel_camera_selection = Toplevel_Camera_Selection(self, self.toplevel_camera_selection_callback)
        
        self.camera = ThreadCamera()
        
        self.build_ui()
        self.FixedUpdate()
    
    def build_ui(self):
        self.configure(padx = 10, pady = 10)
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure((0,2), weight = 0)
        self.grid_columnconfigure(1, weight = 0, minsize = 5)
        
        """ Left """
        self.frame_camera = CTkFrame(self, fg_color = Color().transparent)
        self.frame_camera.grid(row = 0, column = 0, sticky = NS)
        self.frame_camera.grid_columnconfigure(0, weight = 1)
        self.frame_camera.grid_rowconfigure(0, weight = 1)
        self.frame_camera.grid_rowconfigure((1,2), weight = 0)
        
        self.displayCamera = CTkEntry(self.frame_camera, fg_color = Color().black,
                                      width = int(self.camera.width * self.camera.scaledown),
                                      height = int(self.camera.height * self.camera.scaledown),
                                      corner_radius = 0,
                                      justify = CENTER)
        self.displayCamera.insert(0, "Camera Display")
        self.displayCamera.configure(state = "disabled")
        self.displayCamera.grid(row = 0, column = 0)
        self.displayCamera.grid_columnconfigure(0, weight = 1)
        self.displayCamera.grid_rowconfigure(0, weight = 1)
        
        self.showCameraImage_ui = CTkLabel(self.displayCamera, text = "",
                                    fg_color = Color().transparent)
        self.showCameraImage_ui.grid(row = 0, column = 0)
        # self.showCameraImage_ui.place(relx = 0.5, rely = 0.5, anchor = CENTER)
        
        self.frame_program_fps = CTkFrame(self.displayCamera, fg_color = Color().transparent,
                                          corner_radius = 0,
                                          border_width = 0)
        self.frame_program_fps.grid(row = 0, column = 0, sticky = NE)
        self.frame_program_fps.grid_rowconfigure(0, weight = 1)
        self.frame_program_fps.grid_columnconfigure(0, weight = 1)
        
        self.label_program_fps = CTkLabel(self.frame_program_fps, text = "FPS: 0.00",
                                          width = 75,
                                          font = (Font().font, 12),
                                          text_color = Color().white)
        self.label_program_fps.grid(row = 0, column = 0)
        
        self.frame_perspectiveTramform = CTkFrame(self.displayCamera, fg_color = Color().transparent)
        self.frame_perspectiveTramform.grid(row = 0, column = 0, sticky = SE)
        self.frame_perspectiveTramform.grid_rowconfigure(0, weight = 1)
        self.frame_perspectiveTramform.grid_columnconfigure((0,1), weight = 0)
        
        self.btn_reset_perspectiveTranform = CTkButton(self.frame_perspectiveTramform, text = "↻",
                                                        width = 20,
                                                        height = 20,
                                                        font = (Font().font_bold, 16),
                                                        text_color = Color().white,
                                                        corner_radius = 0,
                                                        fg_color = Color().black,
                                                        hover_color = Color().red,
                                                        command = self.btn_resetPerspectiveTransform_callback)
        self.btn_reset_perspectiveTranform.grid(row = 0, column = 0)
        
        self.btn_toggle_perspectiveTranform = CTkButton(self.frame_perspectiveTramform, text = "📐",
                                                        width = 20,
                                                        height = 20,
                                                        font = (Font().font, 16),
                                                        text_color = Color().white,
                                                        corner_radius = 0,
                                                        fg_color = Color().black,
                                                        hover_color = Color().blue,
                                                        command = self.btn_toggle_perspectiveTransform_callback)
        self.btn_toggle_perspectiveTranform.grid(row = 0, column = 1)
        
        self.slider_rotate = CTkSlider(self.frame_camera,
                                    height = 20,
                                    progress_color = Color().blue,
                                    fg_color = Color().blue,
                                    button_color = Color().white,
                                    button_hover_color = Color().white,
                                    from_ = -180, to = 180,
                                    number_of_steps = 720,
                                    command = self.slider_rotate_callback)
        self.slider_rotate.grid(row = 1, column = 0, pady = 10, sticky = EW)
        self.slider_rotate.bind("<Double-Button-1>", lambda event : self.slider_rotate_doubleClick_callback())
        
        self.label_rotate = CTkLabel(self.displayCamera, text = "0°",
                                     fg_color = Color().black,
                                     font = (Font().font_bold, 100),
                                     text_color = Color().white)
        self.label_rotate.grid(row = 0, column = 0)
        self.label_rotate.grid_remove()
        pywinstyles.set_opacity(self.label_rotate, 0.5)
        
        self.frame_OCR = CTkFrame(self.frame_camera, fg_color = Color().transparent)
        self.frame_OCR.grid(row = 2, column = 0, sticky = EW)
        self.frame_OCR.grid_rowconfigure(0, weight = 1)
        self.frame_OCR.grid_columnconfigure(0, weight = 0)
        self.frame_OCR.grid_columnconfigure(1, weight = 1)
        
        self.btn_OCRInspection = CTkButton(self.frame_OCR, text = "OCR",
                                           width = 50,
                                           height = 30,
                                           corner_radius = 0,
                                           border_width = 2,
                                           border_color = Color().white,
                                           hover_color = Color().blue,
                                           fg_color = Color().blue,
                                           font = (Font().font_bold, 20),
                                           command = self.btn_OCRInspection_callback)
        self.btn_OCRInspection.grid(row = 0, column = 0, padx = (0,10))
        
        self.entry_ocrInspection_result = CTkEntry(self.frame_OCR,
                                                   placeholder_text = "OCR Result...",
                                                   height = 30,
                                                   corner_radius = 0,
                                                   border_width = 2,
                                                   border_color = Color().white,
                                                   font = (Font().font, 20))
        self.entry_ocrInspection_result.grid(row = 0, column = 1, sticky = NSEW)
        
        """ Right """
        self.frame_menu_imageProcessing = CTkFrame(self, fg_color = Color().transparent)
        self.frame_menu_imageProcessing.grid(row = 0, column = 2, sticky = NSEW)
        self.frame_menu_imageProcessing.grid_rowconfigure(0, weight = 0)
        self.frame_menu_imageProcessing.grid_rowconfigure(1, weight = 1)
        self.frame_menu_imageProcessing.grid_rowconfigure(2, weight = 0)
        self.frame_menu_imageProcessing.grid_columnconfigure(0, weight = 1)
        
        self.label_menu_imageProcessing_title = CTkLabel(self.frame_menu_imageProcessing, text = "Image Processing Option",
                                                         font = (Font().font_bold, 20),
                                                         fg_color = Color().white,
                                                         text_color = Color().black)
        self.label_menu_imageProcessing_title.grid(row = 0, column = 2, pady = (0, 10), sticky = EW)
        
        self.frame_scrollable_menu_imageProcessing = CTkScrollableFrame(self.frame_menu_imageProcessing, fg_color = Color().transparent,
                                                                        width = 300,
                                                                        corner_radius = 0,
                                                                        scrollbar_button_color = Color().white)
        self.frame_scrollable_menu_imageProcessing.grid(row = 1, column = 2, pady = (0, 10), sticky = NSEW)
        self.frame_scrollable_menu_imageProcessing.grid_columnconfigure(0, weight = 1)
        
        self.module_invert = Module_Invert(self.frame_scrollable_menu_imageProcessing, self)
        self.module_invert.set_callback(self.module_invert_callback)
        self.module_invert.grid(row = 0, column = 0, sticky = EW)
        
        self.module_grayscale = Module_Grayscale(self.frame_scrollable_menu_imageProcessing, self)
        self.module_grayscale.set_callback(self.module_grayscale_callback)
        self.module_grayscale.grid(row = 1, column = 0, sticky = EW)
        
        self.module_mask_rgb = Module_Mask_RGB(self.frame_scrollable_menu_imageProcessing, self)
        self.module_mask_rgb.set_callback(self.module_mask_rgb_callback)
        self.module_mask_rgb.grid(row = 2, column = 0, sticky = EW)
        
        self.module_blur_gussian = Module_Blur_Gaussian(self.frame_scrollable_menu_imageProcessing, self)
        self.module_blur_gussian.set_callback(self.module_blur_gussian_callback)
        self.module_blur_gussian.grid(row = 3, column = 0, sticky = EW)
        
        self.module_mask_gray = Module_Mask_Gray(self.frame_scrollable_menu_imageProcessing, self)
        self.module_mask_gray.set_callback(self.module_mask_gray_callback)
        self.module_mask_gray.grid(row = 4, column = 0, sticky = EW)
        
        self.module_erosion = Module_Erosion(self.frame_scrollable_menu_imageProcessing, self)
        self.module_erosion.set_callback(self.module_erosion_callback)
        self.module_erosion.grid(row = 5, column = 0, sticky = EW)
        
        self.module_dilation = Module_Dilation(self.frame_scrollable_menu_imageProcessing, self)
        self.module_dilation.set_callback(self.module_dilation_callback)
        self.module_dilation.grid(row = 6, column = 0, sticky = EW)
        
        self.module_morph_open = Module_MorphOpen(self.frame_scrollable_menu_imageProcessing, self)
        self.module_morph_open.set_callback(self.module_morph_open_callback)
        self.module_morph_open.grid(row = 7, column = 0, sticky = EW)
        
        self.module_morph_close = Module_MorphClose(self.frame_scrollable_menu_imageProcessing, self)
        self.module_morph_close.set_callback(self.module_morph_close_callback)
        self.module_morph_close.grid(row = 8, column = 0, sticky = EW)
        
        self.module_morph_gradient = Module_MorphGradient(self.frame_scrollable_menu_imageProcessing, self)
        self.module_morph_gradient.set_callback(self.module_morph_gradient_callback)
        self.module_morph_gradient.grid(row = 9, column = 0, sticky = EW)
        
        self.frame_snapshot = CTkFrame(self.frame_menu_imageProcessing, fg_color = Color().transparent)
        self.frame_snapshot.grid(row = 2, column = 2, sticky = EW)
        self.frame_snapshot.grid_rowconfigure(0, weight = 1)
        self.frame_snapshot.grid_columnconfigure((0,2), weight = 1)
        self.frame_snapshot.grid_columnconfigure(1, weight = 0, minsize = 10)
        
        self.btn_cameraSnapshot = CTkButton(self.frame_snapshot, text = "Snapshot",
                                            height = 25,
                                            corner_radius = 0,
                                            border_color = Color().white,
                                            border_width = 2,
                                            fg_color = Color().blue,
                                            font = (Font().font, 16),
                                            text_color = Color().white,
                                            command = self.camera.snapshotImage)
        self.btn_cameraSnapshot.grid(row = 0, column = 0, sticky = EW)
        
        self.btn_processImageSnapshot = CTkButton(self.frame_snapshot, text = "Process Snapshot",
                                                height = 25,
                                                corner_radius = 0,
                                                border_color = Color().white,
                                                border_width = 2,
                                                fg_color = Color().blue,
                                                font = (Font().font, 16),
                                                text_color = Color().white,
                                                command = self.camera.snapshotProcessImage)
        self.btn_processImageSnapshot.grid(row = 0, column = 2, sticky = EW)
        
        self.btn_reset = CTkButton(self.frame_scrollable_menu_imageProcessing, text = "Reset", command = self.resetModule)
        # self.btn_reset.grid(row = 5, column = 2)
        
        self.checkbox_isMask_hsv = CTkCheckBox(self.frame_scrollable_menu_imageProcessing, text = "HSV Mask", command = self.checkbox_mask_hsv_callback)
        # self.checkbox_isMask_hsv.pack()
        
        self.slider_mask_hsv = Slider_MaskHSV(self.frame_scrollable_menu_imageProcessing, self)
        # self.slider_mask_hsv.pack()

    def FixedUpdate(self):
        self.cameraUpdate()
        self.animator_slider_rotate_label_fadeout()
        self.after(10, self.FixedUpdate)

    def combobox_camera_number_callback(self, choice):
        self.camera.changeCameraNumber(int(choice))

    def slider_rotate_callback(self, value):
        self.camera.rotateAngle = value
        self.label_rotate.configure(text = f"{self.camera.rotateAngle}°")
        self.slider_rotate_label_fadeout()
        
    def slider_rotate_doubleClick_callback(self):
        self.slider_rotate.set(0)
        self.camera.rotateAngle = 0
        self.label_rotate.configure(text = f"{self.camera.rotateAngle}°")
        self.slider_rotate_label_fadeout()
    
    def slider_rotate_label_fadeout(self):
        self.animation_slider_label_timer = 200
        self.animation_silder_label_isplay = True
        self.label_rotate.grid()
    
    def animator_slider_rotate_label_fadeout(self):
        if self.animation_silder_label_isplay:
            self.animation_slider_label_timer -= 10
            if self.animation_slider_label_timer < 0:
                self.animation_silder_label_isplay = False
                self.label_rotate.grid_remove()
    
    def checkbox_mask_hsv_callback(self):
        if self.checkbox_isMask_hsv.get():
            self.camera.isMask_hsv = True
        else:
            self.camera.isMask_hsv = False
    
    def slider_mask_hsv_callback(self):
        self.camera.mask_hsv_upper = self.slider_mask_hsv.upper_range
        self.camera.mask_hsv_lower = self.slider_mask_hsv.lower_range
    
    def module_invert_callback(self):
        self.camera.isInvert = self.module_invert.isActive
        
        if self.camera.isInvert:
            self.camera.add_imageProcessing(self.camera.imageProcessing_invert)
        else:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_invert)
    
    def module_grayscale_callback(self):
        self.camera.isGray = self.module_grayscale.grayscale_isActive
        self.camera.isGrayChannel = self.module_grayscale.grayscale_channel_isActive
        self.camera.gray_channel = self.module_grayscale.grayscale_channel
        
        if self.camera.isGray:
            self.camera.add_imageProcessing(self.camera.imageProcessing_grayscale)
        elif not self.camera.isGray:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_grayscale)
            
        if self.camera.isGrayChannel:
            self.camera.add_imageProcessing(self.camera.imageProcessing_grayscale_channel)
        elif not self.camera.isGrayChannel:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_grayscale_channel)
    
    def module_mask_rgb_callback(self):
        self.camera.mask_red = self.module_mask_rgb.mask_red
        self.camera.mask_green = self.module_mask_rgb.mask_green
        self.camera.mask_blue = self.module_mask_rgb.mask_blue
        
        if self.camera.mask_red:
            self.camera.add_imageProcessing(self.camera.imageProcessing_mask_red)
        elif not self.camera.mask_red:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_mask_red)
        
        if self.camera.mask_green:
            self.camera.add_imageProcessing(self.camera.imageProcessing_mask_green)
        elif not self.camera.mask_green:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_mask_green)
        
        if self.camera.mask_blue:
            self.camera.add_imageProcessing(self.camera.imageProcessing_mask_blue)
        elif not self.camera.mask_blue:
            self.camera.remove_imageProcessing(self.camera.imageProcessing_mask_blue)
            
    
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
            self.showCameraImage_ui.configure(image = CTkImage(self.processImage, 
                                                               size = (int(self.camera.width * self.camera.scaledown), self.camera.height * self.camera.scaledown)))
            self.label_program_fps.configure(text = f"FPS: {self.camera.fps}")
        except:
            pass
    
    def resetModule(self):
        self.camera.reset_imageProcessing()
        self.module_grayscale.reset()
        self.module_mask_rgb.reset()
    
    def btn_toggle_perspectiveTransform_callback(self):
        self.camera.isPerspectiveTransform ^= 1
        
        # if self.checkbox_isPerspectiveTransform.get():
        #     self.camera.isPerspectiveTransform = True
        # else:
        #     self.camera.isPerspectiveTransform = False
    
    def btn_resetPerspectiveTransform_callback(self):
        self.camera.isPerspectiveTransform = False
        self.camera.isPerspectivePointSet = False
        
    def btn_OCRInspection_callback(self):
        self.camera.selectOCRInspection()
        self.entry_ocrInspection_result.delete(0, END)
        self.entry_ocrInspection_result.insert(1, self.camera.ocrResult)
    
    def toplevel_camera_selection_callback(self):
        self.camera.cameraSetup(self.toplevel_camera_selection.cameraNumber)
        self.camera.width = self.toplevel_camera_selection.cameraResolution[0]
        self.camera.height = self.toplevel_camera_selection.cameraResolution[1]
        self.camera.isSelectCamera = True
        self.camera.start()
        
        self.displayCamera.configure(width = self.camera.width * self.camera.scaledown,
                                     height = self.camera.height * self.camera.scaledown)
        self.slider_rotate.configure(width = self.camera.width * self.camera.scaledown)
    
    def program_keyboard_command(self, event):
        # Show/Hide Camera FPS
        if event.keysym == "F1":
            if self.frame_program_fps.grid_info():
                self.frame_program_fps.grid_remove()
            else:
                self.frame_program_fps.grid()
            
    def exitApplication(self):
        if self.camera.isSelectCamera:
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

class Module_Invert(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.isActive = False
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Invert",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure(0, weight = 0)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_checkbox = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_checkbox.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_checkbox.grid_columnconfigure((0,2), weight = 0)
        self.frame_checkbox.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_checkbox.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_invert = CTkCheckBox(self.frame_checkbox, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_invert_callback)
        self.checkbox_invert.grid(row = 0, column = 0)
        
        self.label_invert_title = CTkLabel(self.frame_checkbox, text = "Invert",
                                      height = 15)
        self.label_invert_title.grid(row = 0, column = 2, sticky = W)
    
    def checkbox_invert_callback(self):
        if self.checkbox_invert.get():
            self.isActive = True
        else:
            self.isActive = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
        
    def reset(self):
        self.isActive = False
        
        if self.callback:
            self.callback()

class Module_Grayscale(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.grayscale_isActive = False
        self.grayscale_channel_isActive = False
        self.grayscale_channel = 0
        self.callback = None
        
        self.channel = ["R", "G", "B"]
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Grayscale",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure(0, weight = 0)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_checkbox = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_checkbox.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_checkbox.grid_columnconfigure((0,2), weight = 0)
        self.frame_checkbox.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_checkbox.grid_rowconfigure(0, weight = 1)
        
        self.checkbox_grayscale = CTkCheckBox(self.frame_checkbox, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_grayscale_callback)
        self.checkbox_grayscale.grid(row = 0, column = 0)
        
        self.label_grayscale_title = CTkLabel(self.frame_checkbox, text = "Grayscale",
                                      height = 15)
        self.label_grayscale_title.grid(row = 0, column = 2, sticky = W)
        
        self.checkbox_grayscale_channel = CTkCheckBox(self.frame_checkbox, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_grayscale_channel_callback)
        self.checkbox_grayscale_channel.grid(row = 1, column = 0)
        
        self.label_grayscale_channel_title = CTkLabel(self.frame_checkbox, text = "Channel Grayscale",
                                      height = 15)
        self.label_grayscale_channel_title.grid(row = 1, column = 2, sticky = W)
        
        self.segment_btn_grayscale_channel = CTkSegmentedButton(self.frame_checkbox, values = self.channel,
                                                                command = self.segment_btn_grayscale_channel_callback)
        self.segment_btn_grayscale_channel.set(self.channel[0])
        self.segment_btn_grayscale_channel.grid(row = 2, column = 2, sticky = W)
    
    def checkbox_grayscale_callback(self):
        if self.checkbox_grayscale.get():
            self.grayscale_isActive = True
        else:
            self.grayscale_isActive = False
        
        if self.callback:
            self.callback()
    
    def checkbox_grayscale_channel_callback(self):
        if self.checkbox_grayscale_channel.get():
            self.grayscale_channel_isActive = True
        else:
            self.grayscale_channel_isActive = False
        
        if self.callback:
            self.callback()
    
    def segment_btn_grayscale_channel_callback(self, value):
        self.grayscale_channel = self.channel.index(value)
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
        
    def reset(self):
        self.grayscale_isActive = False
        self.grayscale_channel_isActive = False
        self.checkbox_grayscale.deselect()
        self.checkbox_grayscale_channel.deselect()
        
        if self.callback:
            self.callback()

class Module_Mask_RGB(CTkFrame):
    def __init__(self, master, mainApp, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.mainApp = mainApp
        self.mask_red = False
        self.mask_green = False
        self.mask_blue = False
        self.callback = None
        
        self.configure(fg_color = Color().transparent)
        
        self.build_ui()
    
    def build_ui(self):
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "RGB Mask",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
        self.frame_border.grid_rowconfigure(0, weight = 0)
        self.frame_border.grid_columnconfigure(0, weight = 1)
        
        self.frame_checkbox = CTkFrame(self.frame_border, fg_color = Color().transparent)
        self.frame_checkbox.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = W)
        self.frame_checkbox.grid_columnconfigure((0,2), weight = 0)
        self.frame_checkbox.grid_columnconfigure((1), weight = 0, minsize = 5)
        self.frame_checkbox.grid_rowconfigure((0,1,2), weight = 0)
        
        self.checkbox_red = CTkCheckBox(self.frame_checkbox, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_red_callback)
        self.checkbox_red.grid(row = 0, column = 0)
        
        self.label_red_title = CTkLabel(self.frame_checkbox, text = "Mask Red",
                                      height = 15)
        self.label_red_title.grid(row = 0, column = 2, sticky = W)
        
        self.checkbox_green = CTkCheckBox(self.frame_checkbox, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_green_callback)
        self.checkbox_green.grid(row = 1, column = 0)
        
        self.label_green_title = CTkLabel(self.frame_checkbox, text = "Mask Green",
                                      height = 15)
        self.label_green_title.grid(row = 1, column = 2, sticky = W)
        
        self.checkbox_blue = CTkCheckBox(self.frame_checkbox, text = "",
                                           width = 15, height = 15,
                                           command = self.checkbox_blue_callback)
        self.checkbox_blue.grid(row = 2, column = 0)
        
        self.label_blue_title = CTkLabel(self.frame_checkbox, text = "Mask Blue",
                                      height = 15)
        self.label_blue_title.grid(row = 2, column = 2, sticky = W)
    
    def checkbox_red_callback(self):
        if self.checkbox_red.get():
            self.mask_red = True
        else:
            self.mask_red = False
        
        if self.callback:
            self.callback()
    
    def checkbox_green_callback(self):
        if self.checkbox_green.get():
            self.mask_green = True
        else:
            self.mask_green = False
        
        if self.callback:
            self.callback()
            
    def checkbox_blue_callback(self):
        if self.checkbox_blue.get():
            self.mask_blue = True
        else:
            self.mask_blue = False
        
        if self.callback:
            self.callback()
    
    def set_callback(self, callback):
        self.callback = callback
        
    def reset(self):
        self.mask_red = False
        self.checkbox_red.deselect()
        self.mask_green = False
        self.checkbox_green.deselect()
        self.mask_blue = False
        self.checkbox_blue.deselect()
        
        if self.callback:
            self.callback()

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
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Gaussian Blur",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
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
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Grayscale Mask",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
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
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Erosion",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
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
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Dilation",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
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
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Morphological Opening",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
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
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 100, number_of_steps = 100)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 100, number_of_steps = 100)
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
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Morphological Closing",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
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
        self.slider_kernel.slider_kernelx.configure(from_ = 1, to = 100, number_of_steps = 100)
        self.slider_kernel.slider_kernely.configure(from_ = 1, to = 100, number_of_steps = 100)
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
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_module_title = CTkFrame(self, fg_color = Color().white,
                                           corner_radius = 0)
        self.frame_module_title.grid(row = 0, column = 0, ipady = 2, sticky = EW)
        self.frame_module_title.grid_columnconfigure(0, weight = 1)
        self.frame_module_title.grid_rowconfigure(0, weight = 1)
        
        self.label_module_title = CTkLabel(self.frame_module_title, text = "Morphological Gradient",
                                           height = 15,
                                           text_color = Color().black,
                                           font = Font().module_label)
        self.label_module_title.grid(row = 0, column = 0)
        
        self.frame_border = CTkFrame(self, fg_color = Color().transparent,
                                      corner_radius = 0, 
                                      border_width = 1, 
                                      border_color = Color().white)
        self.frame_border.grid(row = 1, column = 0, ipadx = 10, ipady = 10, sticky = NSEW)
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
        self.frame_main.grid(row = 0, column = 0, sticky = EW) 
        self.frame_main.grid_columnconfigure((0,1), weight = 1)
        self.frame_main.grid_columnconfigure(2, weight = 1, minsize = 30)
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
        self.frame_main.grid_columnconfigure((0,1), weight = 1)
        self.frame_main.grid_columnconfigure(2, weight = 1, minsize = 30)
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

class Toplevel_Camera_Selection(CTkToplevel):
    def __init__(self, master, callback, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        
        # Variable
        self.cameraNumber = 0
        self.cameraResolution = (2592, 1944)
        
        self.isCameraSelect = False
        self.callback = callback
        
        self.camera_resolution_list = ["1920x1080", "2592x1944"]
        
        # Setting
        self.attributes("-topmost", False)  # Always on top
        self.protocol("WM_DELETE_WINDOW", self.quit)
        
        screen_width = 400
        screen_height = 250
        self.geometry(f"{screen_width}x{screen_height}+{960-int(screen_width/2)}+{540-int(screen_height/2)}")
        
        self.title("Select Camera")
        
        self.resizable(False,False)
        
        """ Main """
        self.build_ui()
        self.bringToTop()

    def build_ui(self):
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        
        self.frame_main = CTkFrame(self, fg_color = Color().transparent)
        self.frame_main.grid_rowconfigure(0, weight = 1)
        self.frame_main.grid_columnconfigure(0, weight = 1)
        self.frame_main.grid(row = 0, column = 0, sticky = NSEW)

        self.frame_camera_selection = CTkFrame(self.frame_main, fg_color = Color().transparent)
        self.frame_camera_selection.grid_rowconfigure(0, weight = 0)
        self.frame_camera_selection.grid_columnconfigure(0, weight = 1)
        self.frame_camera_selection.grid(row = 0, column = 0)
        
        self.label_camera_number_title = CTkLabel(self.frame_camera_selection, text = "Camera Number",
                                                  text_color = Color().white)
        self.label_camera_number_title.grid(row = 0, column = 0, sticky = W)
        
        camera_number_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        self.combobox_camera_number = CTkComboBox(self.frame_camera_selection, values = camera_number_list,
                                                  command = self.combobox_camera_number_callback)
        self.combobox_camera_number.set("0")
        self.combobox_camera_number.grid(row = 1, column = 0, pady = (0, 20), sticky = EW)
        
        self.label_camera_resolution_title = CTkLabel(self.frame_camera_selection, text = "Camera Resulution",
                                                  text_color = Color().white)
        self.label_camera_resolution_title.grid(row = 2, column = 0, sticky = W)
        
        self.combobox_camera_resolution = CTkComboBox(self.frame_camera_selection, values = self.camera_resolution_list,
                                                  command = self.combobox_camera_resolution_callback)
        self.combobox_camera_resolution.set("2592x1944")
        self.combobox_camera_resolution.grid(row = 3, column = 0, pady = (0, 20), sticky = EW)
        
        self.btn_ok = CTkButton(self.frame_camera_selection, text = "OK",
                                width = 100, height = 50,
                                corner_radius = 0,
                                border_width = 0,
                                command = self.btn_ok_callback)
        self.btn_ok.grid(row = 4, column = 0)
    
    def bringToTop(self):
        self.after(100, lambda: self.attributes("-topmost", True))
        self.after(400, lambda: self.attributes("-topmost", False))
    
    def combobox_camera_number_callback(self, choice):
        self.cameraNumber = choice
    
    def combobox_camera_resolution_callback(self, choice):
        resulution = choice.split("x")
        self.cameraResolution = (int(resulution[0]), int(resulution[1]))
    
    def btn_ok_callback(self):
        if self.cameraNumber == None:
            return
        
        self.isCameraSelect = True
        if self.callback:
            self.callback()
        self.destroy()
    
    def quit(self):
        if not self.isCameraSelect:
            self.master.exitApplication()
            
app = MainApp()
app.mainloop()