###
###
###
import cv2
import numpy as np
import face_recognition
import dlib
import time
def draw_flip_rectangle(img,x,y,w,h,rec_size):
    size=img.shape
    width=size[1]
    cv2.rectangle(img,(width-x,y),(width-(x+w),y+h),(255,0,0),rec_size)

def find_face_in_batchs_in_video(filename):
    # This code finds all faces in a list of images using the CNN model.
    #
    # This demo is for the _special case_ when you need to find faces in LOTS of images very quickly and all the images
    # are the exact same size. This is common in video processing applications where you have lots of video frames
    # to process.
    #
    # If you are processing a lot of images and using a GPU with CUDA, batch processing can be ~3x faster then processing
    # single images at a time. But if you aren't using a GPU, then batch processing isn't going to be very helpful.
    #
    # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read the video file.
    # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
    # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

    # Open video file
    video_capture = cv2.VideoCapture(filename)

    frames = []
    frame_count = 0

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)

        # Every 25 frames (the default batch size), batch process the list of frames to find faces
        if len(frames) == 1:
            batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

            # Now let's list all the faces we found in all 128 frames
            for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                number_of_faces_in_frame = len(face_locations)

                frame_number = frame_count - 1 + frame_number_in_batch
                print("I found {} face(s) in frame #{}.".selfat(number_of_faces_in_frame, frame_number))

                for face_location in face_locations:
                    # Print the location of each face in this frame
                    top, right, bottom, left = face_location
                    cv2.imshow('face',frame[top-10:bottom+10,left-10:right+10])
                    print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".selfat(top,
                                                                                                                   left,
                                                                                                                   bottom,
                                                                                                                   right))
                    cv2.waitKey(0)

            # Clear the frames array to start the next batch
            frames = []

def face_detection_demo(filename):
    t0=time.time()
    img=cv2.imread(filename)
    face_locations=face_recognition.face_locations(img,model='cnn')
    for facelocation in face_locations:
        top,right,bottom,left=facelocation
        cv2.rectangle(img,(left,bottom),(right,top),(255,0,0),2)
    t1=time.time()
    print(str(t1-t0)+'s')
    cv2.imshow('faces',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#
def face_detection_in_image_20faces(img,profile,frontal,profile_scale,profile_nei,frontal_scale,frontal_nei):
    if frontal:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')  #
    if profile:
        profileface_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')  #
    t0 = time.time()
    #img = cv2.resize(img, (3200, 1800))
    cv2.namedWindow('video')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grey=cv2.erode(grey,(3,3))
    # grey=cv2.dilate(grey,(3,3))
    grey=cv2.equalizeHist(grey)   #
    flip = cv2.flip(img, 1)  
    grey_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)
    if frontal:
        faces = face_cascade.detectMultiScale(grey,frontal_scale/float(100), frontal_nei)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if profile:
        profileface_right = profileface_cascade.detectMultiScale(grey_flip, profile_scale/float(100),profile_nei)  #
        profileface = profileface_cascade.detectMultiScale(grey, profile_scale/float(100),profile_nei)
        for (x, y, w, h) in profileface:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
        for (x, y, w, h) in profileface_right:
            draw_flip_rectangle(img, x, y, w, h, 2)
    img = cv2.resize(img, (1000, 532))
    cv2.imshow('video', img)
    t1 = time.time()
    print('time:', str(t1 - t0))
    #cv2.imwrite(str(filename)+'detected.jpg',img)      







def face_detection_with_dlib(img):
    detector=dlib.get_frontal_face_detector()   #打开分类器
    t0=time.time()
    b,g,r=cv2.split(img)    #分离三个通道
    img2=cv2.merge([r,g,b]) #重新生成图片
    dets=detector(img,1)
    t1=time.time()
    print(str(t1-t0)+'s')
    # 在图片中标注人脸，并显示
    for index,face in enumerate(dets):
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)


def nothing(x):#滑动条回调函数
    pass
def face_detection_with_hog_and_haar(filename):

    img=cv2.imread(filename,cv2.IMREAD_COLOR)
    face_detection_with_dlib(img)
    face_detection_in_image_20faces(img)
def face_detection_with_hog_and_haar_in_video(filename):
    cap=cv2.VideoCapture(filename)
    control = np.zeros((300,512, 3), np.uint8)
    cv2.namedWindow('control')
    cv2.createTrackbar('haar','control',0,1,nothing)
    cv2.createTrackbar('dlib','control',0,1,nothing)
    cv2.createTrackbar('harr_frontal','control',0,1,nothing)
    cv2.createTrackbar('harr_frontal_scale','control',0,30,nothing)
    cv2.createTrackbar('haar_frontal_nei','control',2,25,nothing)
    cv2.createTrackbar('harr_profile','control',0,1,nothing)
    cv2.createTrackbar('harr_profile_scale','control',0,30,nothing)
    cv2.createTrackbar('haar_profile_nei','control',2,25,nothing)
    cv2.createTrackbar('resize','control',0,100,nothing)
    while 1:
        cv2.imshow('control',control)
        haar=cv2.getTrackbarPos('haar','control')
        dlib=cv2.getTrackbarPos('dlib','control')
        haar_frontal=cv2.getTrackbarPos('harr_frontal','control')
        haar_frontal_scale=cv2.getTrackbarPos('harr_frontal_scale','control')
        haar_frontal_nei=cv2.getTrackbarPos('haar_frontal_nei','control')
        harr_profile=cv2.getTrackbarPos('harr_profile','control')
        harr_profile_scale=cv2.getTrackbarPos('harr_profile_scale','control')
        harr_profile_nei=cv2.getTrackbarPos('haar_profile_nei','control')
        resizepara=cv2.getTrackbarPos('resize','control')
        ret,img=cap.read()          #读取帧
        if resizepara>0:
            cv2.resize(img,(480+16*resizepara,270+9*resizepara))
        if dlib:
            face_detection_with_dlib(img)
        if haar:
            face_detection_in_image_20faces(img,harr_profile,haar_frontal,harr_profile_scale+111,harr_profile_nei,haar_frontal_scale+111,haar_frontal_nei)
        cv2.waitKey(1)&0xFF==ord('q')
    cv2.destroyAllWindows()



def face_detector_withresize_and_write(img,face_cascade,profileface_cascade,frontscale,frontnei,profilescale,profilenei,resizeface,pad):
    t0=time.time()
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grey=cv2.equalizeHist(grey)         #直方图均衡化，可以去除该步骤
    grey_flip=cv2.flip(grey,1)      #翻转图像
    profileface_right=profileface_cascade.detectMultiScale(grey_flip,profilescale,profilenei)#将左脸分类器通过翻转图像的方法建立为右脸分类器
    faces=face_cascade.detectMultiScale(grey,frontscale,frontnei)
    profileface=profileface_cascade.detectMultiScale(grey,profilescale,profilenei)

    for (x,y,w,h) in faces:
        facename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'    #文件名为人脸所在的坐标
        part=img[y-pad:y+h+pad,x-pad:x+w+pad]                   #边缘填充后的矩阵
        part=cv2.resize(part,resizeface)                        #提取人脸并且进行resize
        cv2.imwrite(facename,part)                              #写成文件的形式保存
    for (x,y,w,h) in profileface:
        profilename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'
        part1=img[y-pad:y+h+pad,x-pad:x+w+pad]
        part1=cv2.resize(part1,resizeface)
        cv2.imwrite(profilename,part1)
    for (x,y,w,h) in profileface_right:
        Rprofilename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'
        size=img.shape
        width=size[1]
        part2=img[y-pad:y+h+pad,width-(x+w)-pad:width-x+pad]
        part2=cv2.resize(part2,resizeface)
        cv2.imwrite(Rprofilename,part2)
    t1=time.time()
    print('time:',str(t1-t0))



def face_detector_with_img_out(img,face_cascade,profileface_cascade,frontscale,frontnei,profilescale,profilenei,resizeface,pad):
    detected_faces=[]
    t0=time.time()
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grey=cv2.equalizeHist(grey)         #直方图均衡化，可以去除该步骤
    grey_flip=cv2.flip(grey,1)      #翻转图像
    profileface_right=profileface_cascade.detectMultiScale(grey_flip,profilescale,profilenei)#将左脸分类器通过翻转图像的方法建立为右脸分类器
    faces=face_cascade.detectMultiScale(grey,frontscale,frontnei)
    profileface=profileface_cascade.detectMultiScale(grey,profilescale,profilenei)
    for (x,y,w,h) in faces:
        facename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'    #文件名为人脸所在的坐标
        part=img[y-pad:y+h+pad,x-pad:x+w+pad]                   #边缘填充后的矩阵
        part=cv2.resize(part,resizeface)                        #提取人脸并且进行resize
        detected_faces.append(part)
    for (x,y,w,h) in profileface:
        profilename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'
        part1=img[y-pad:y+h+pad,x-pad:x+w+pad]
        part1=cv2.resize(part1,resizeface)
        detected_faces.append(part1)
    for (x,y,w,h) in profileface_right:
        Rprofilename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'
        size=img.shape
        width=size[1]
        part2=img[y-pad:y+h+pad,width-(x+w)-pad:width-x+pad]
        part2=cv2.resize(part2,resizeface)
        detected_faces.append(part2)
    t1=time.time()
    print('time:',str(t1-t0))
    return detected_faces


def face_detector_with_img_out_frontal(img,face_cascade,frontscale,frontnei,resizeface,pad):
    detected_faces=[]
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grey=cv2.equalizeHist(grey)         #直方图均衡化，可以去除该步骤
    faces=face_cascade.detectMultiScale(grey,frontscale,frontnei)
    faces_axi=[]
    for (x,y,w,h) in faces:
        faces_axi.append([x,y])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        facename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'    #文件名为人脸所在的坐标
        if y-pad<0 or x-pad<0 or y+h+pad>img.shape[0] or x+w+pad>img.shape[1]:
            continue
        part=img[y-pad:y+h+pad,x-pad:x+w+pad]                   #边缘填充后的矩阵
        part=cv2.resize(part,resizeface)                        #提取人脸并且进行resize
        detected_faces.append(part)
    return detected_faces,faces_axi


