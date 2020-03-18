import numpy as np 
from cv2 import cv2 
from matplotlib import pyplot as plt
import glob
import json
#标定单目
def calibrateOne(img_path,cameraParam):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,30,0.001)
    #设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    #获取标定板角点的位置
    objp = np.zeros((4*7,3),np.float32)
    objp[:,:2] = np.mgrid[0:7,0:4].T.reshape(-1,2)#将世界坐标系建在标定板上

    obj_points = []#3D points
    img_points = []#2D points
    
    images = glob.glob(img_path)
    i = 0
    for fname in images:
        img = cv2.imread(fname)
        #print(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        h, w = gray.shape[:2]
        ret,corners = cv2.findChessboardCorners(gray,(7,4),None)
        if ret:
            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            #在原角点的基础上寻找亚像素点
            #print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img,(7,4),corners,ret)
            i+=1
            
        
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(obj_points,img_points,size,None,None,criteria)
        #计算误差
    newcameramtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
    #dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
    #x,y,w,h = roi
    
    cameraParam['ret'] = ret
    cameraParam['mtx'] = mtx
    cameraParam['dist'] = dist
    cameraParam['rvecs'] = rvecs
    cameraParam['tvecs'] = tvecs
    cameraParam['img_points'] = img_points
    cameraParam['obj_points'] = obj_points
    cameraParam['size'] = size
    cameraParam['newmtx'] = newcameramtx
    print("Calibrate One Camera Success")

def rectify_pair(image_left,image_right,viz=False):
    #特征点匹配
    grayL = cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(grayL, None)
    kp2, des2 = surf.detectAndCompute(grayR, None)
    img = cv2.drawKeypoints(grayL,kp1,image_left)
    cv2.imshow("keyPointsOfLeft",img)
    bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    # 特征描述子匹配
    matches=bf.match(des1,des2)
    points1 = []
    points2 = []
    for match in matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)
    #matches=sorted(matches,key=lambda x:x.distance)
    # print(len(matches))
    img3=cv2.drawMatches(grayL,kp1,grayR,kp2,matches[:20],None,flags=2)
    cv2.imshow('matches',img3)

    # find the fundamental matrix
    F, mask = cv2.findFundamentalMat(np.array(points1), np.array(points2), cv2.RANSAC, 3, 0.99)

    # rectify the images, produce the homographies: H_left and H_right
    retval, H_left, H_right = cv2.stereoRectifyUncalibrated(np.array(points1), np.array(points2), F, image_left.shape[:2])

    return F, H_left, H_right
def calibrateTwo(leftCameraParam,rightCameraParam,lmap,rmap,leftImg,rightImg):
    img1 = cv2.imread(leftImg)
    img2 = cv2.imread(rightImg)
    F,H_left,H_right = rectify_pair(img1,img2,True)
    R1 = np.linalg.inv(leftCameraParam['mtx'])*H_left*leftCameraParam['mtx']
    R2 = np.linalg.inv(rightCameraParam['mtx'])*H_right*rightCameraParam['mtx']
    rms, C1, dist1, C2, dist2, R, T, E,F = cv2.stereoCalibrate(leftCameraParam['obj_points'], leftCameraParam['img_points'], rightCameraParam['img_points'], leftCameraParam['mtx'], leftCameraParam['dist'],rightCameraParam['mtx'], rightCameraParam['dist'],leftCameraParam['size'],flags=cv2.CALIB_USE_INTRINSIC_GUESS )
    
    Q = cv2.stereoRectify(C1,dist1,C2,dist2,leftCameraParam['size'],R,T,alpha = 1)
    #Computes rectification transforms for each head of a calibrated stereo camera.
    left_map1, left_map2 = cv2.initUndistortRectifyMap(C1, dist1, R1*R2, leftCameraParam['newmtx'], leftCameraParam['size'], cv2.CV_16SC2)
    #计算矫正参数
    right_map1, right_map2 = cv2.initUndistortRectifyMap(C2, dist2, R1*R2, rightCameraParam['newmtx'], leftCameraParam['size'], cv2.CV_16SC2)
    print('calibrate two camera success')
    lmap['lm1'] = left_map1
    lmap['lm2'] = left_map2
    rmap['rm1'] = right_map1
    rmap['rm2'] = right_map2
    lmap['Q'] = Q
    lmap['R1'] = R1
    rmap['R2'] = R2
    cv2.waitKey(-1)

#生成深度图
def depth(lmap,rmap,lImg,rImg,f):
    l_images = cv2.imread(lImg)
    #cv2.imshow('left image',l_images)
    r_images = cv2.imread(rImg)
    #cv2.imshow('right image',r_images)
    
    #矫正双目
    img1_rectified = cv2.remap(l_images, lmap['lm1'], lmap['lm2'], cv2.INTER_LINEAR)
    #left_map1&left_map2=>重映射1/2 采用线性插值
    img2_rectified = cv2.remap(r_images, rmap['rm1'], rmap['rm2'], cv2.INTER_LINEAR)

    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # calculate histogram
    cv2.imshow("left",imgL)
    cv2.imshow("right",imgR)
    window_size = 3
    minDisp = 0
    numDisp = 128 - minDisp
    stereo = cv2.StereoSGBM_create(minDisparity = minDisp,
        numDisparities = numDisp,
        blockSize = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 50,
        speckleRange = 16
    )
    #stereo = cv2.StereoBM_create(numDisparities=0, blockSize=5)
    #disparity = stereo.compute(imgL, imgR)
    #disparity = stereo.compute(imgGrayL, imgGrayR).astype(np.float32) / 16
    #disp = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #disparity = (disparity - minDisp) / numDisp
    #threeD_disp = cv2.reprojectImageTo3D(disparity,lmap['Q'],0)
    #f.write(str(threeD_disp))
    #cv2.imshow('3d',threeD_disp)
    f.write(str(lmap))
    f.write(str(rmap))
    
#def WorldPos(leftCameraParam,rightCameraParam,lmap)    

left_caliPath = "E:/DC/CaptrueData2019.12.11/left1/*.jpg"
right_caliPath = "E:/DC/CaptrueData2019.12.11/right1/*.jpg"
lImg = "E:/DC/CaptrueData2019.12.11/left1/20191211_080127_650__1_27.662.jpg"
rImg = "E:/DC/CaptrueData2019.12.11/right1/20191211_080127_650__0_27.678.jpg"
f = open('./CameraParam.txt','w')
f_disp = open('./disparity.txt','w')
left_CameraParam = {}
right_CameraParam = {}
l_map = {}
r_map = {}
calibrateOne(left_caliPath,left_CameraParam)
calibrateOne(right_caliPath,right_CameraParam)
#f.write(str(left_CameraParam))
#f.write(str(right_CameraParam))
calibrateTwo(left_CameraParam,right_CameraParam,l_map,r_map,lImg,rImg)
depth(l_map,r_map,lImg,rImg,f_disp)
f.close()
f_disp.close()