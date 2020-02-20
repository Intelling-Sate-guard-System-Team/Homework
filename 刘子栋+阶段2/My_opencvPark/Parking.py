# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import cv2
import os,glob
import numpy as np


class Parking:

    def show_images(self,images,cmap=None):
        cols = 2
        rows = (len(images)+1)//cols

        plt.figure(figsize=(15,12))
        for i,image in enumerate(images):
            plt.subplot(rows,cols,i+1)
            cmap = 'gray' if len(image.shape) == 2 else cmap
            plt.imshow(image,cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0,h_pad=0,w_pad=0)  # 可以使x,y轴的labels不被遮盖
        plt.show()

    def cv_show(self,name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def select_rgb_white_yellow(self,image):
        # 设置最小和最大阈值，过滤掉背景
        lower = np.uint8([120,120,120])
        upper = np.uint8([255,255,255])
        # cv2.inRange()会把image(lower,upper)范围以外的部分全部变为0，在这之间的变成255
        white_mask =  cv2.inRange(image,lower,upper)
        self.cv_show('white_mask',white_mask)
        # 利用掩膜（mask）进行“与”操作，即掩膜图像白色区域是对需要处理图像像素的保留，
        # 黑色区域是对需要处理图像像素的剔除，其余按位操作原理类似只是效果不同而已
        masked = cv2.bitwise_and(image,image,mask=white_mask)
        self.cv_show('masked',masked)
        return masked

    def convert_gray_scale(self, image):
        #灰度图
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        #边缘检测
        return cv2.Canny(image,low_threshold,high_threshold)

    def filter_region(self, image, vertices):
        #去除不需要的地方
        mask = np.zeros_like(image)
        if len(mask.shape) == 2:
            # 根据顶点位置，把顶点覆盖的内部区域填充起来
            cv2.fillPoly(mask,vertices,255)
            self.cv_show('mask',mask)
        #保留image中mask的区域,其它区域过滤掉
        return cv2.bitwise_and(image,mask)

    def select_region(self, image):
        #手动选择区域
        #先设置好要选定的区域的顶点，由这些顶点来连接成一个区域
        rows, cols = image.shape[:2]
        rows, cols = image.shape[:2]
        pt_1 = [cols * 0.05, rows * 0.90]  # 左下1
        pt_2 = [cols * 0.05, rows * 0.70]  # 左下_上2
        pt_3 = [cols * 0.30, rows * 0.55]  # 中3
        pt_4 = [cols * 0.6, rows * 0.15]   # 中上4
        pt_5 = [cols * 0.90, rows * 0.15]  # 右上5
        pt_6 = [cols * 0.90, rows * 0.90]  # 右下6
        # 转化为3维的矩阵
        vertices = np.array([[pt_1,pt_2,pt_3,pt_4,pt_5,pt_6]],dtype=np.int32)
        point_img = image.copy()
        #灰度图还原为彩色图
        point_img = cv2.cvtColor(point_img,cv2.COLOR_GRAY2BGR)
        #从顶点中获取x,y坐标并画出顶点
        for p in vertices[0]:
            cv2.circle(point_img,(p[0],p[1]),10,(0,0,255),4)
        self.cv_show('point_img',point_img)
        
        return self.filter_region(image,vertices)

    def hough_lines(self, image):
        # 输入的图像是边缘检测后的结果
        # minLineLengh(线的最短长度，比这个短的都被忽略)和MaxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
        # rho距离精度,theta角度精度,越小越精确,threshod超过设定阈值才被检测出线段,越小检测的直线越多
        return cv2.HoughLinesP(image,rho=0.1,theta=np.pi/10,threshold=15, minLineLength=9, maxLineGap=4)

    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        if make_copy:
            image = np.copy(image)
        # 过滤霍夫变换检测到的直线
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                # 设置满足情况的条件：线是水平的，线的长度是有限的
                if abs(y2-y1) <= 1 and abs(x2-x1) >= 25 and abs(x2-x1) <= 55:
                    cleaned.append((x1,y1,x2,y2))
                    cv2.line(image,(x1,y1),(x2,y2),color,thickness)
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)

        #Step 1:过滤部分直线
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        #Step 2:对直线按照x1,y1进行排序
        import operator
        list1 = sorted(cleaned,key=operator.itemgetter(0,1))

        #Step 3:找到多个列，相当于每列是一排车
        clusters = {}
        dIndex = 0
        clus_dist = 10 # 是否为同一列的阈值
        for i in range(len(list1) - 1):
            distance = abs(list1[i+1][0] - list1[i][0])  # 两个直线的x1之间的距离
            #如果距离小于阈值，则为同一列
            if distance <= clus_dist:
                if not dIndex in clusters.keys():
                    clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i+1])
            else:
                dIndex += 1

        #Step 4: 得到列的左上坐标和右下坐标，并根据2个坐标画出矩形框
        rects = {}
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 5:
                #由于所有直线的x1,x2,y1,y2可能有所差别，但差别不大，所以取其平均值作为坐标点
                cleaned = sorted(cleaned,key=lambda tup:tup[1])
                avg_y1 = cleaned[0][1]
                avg_y2 = cleaned[-1][1]
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1/len(cleaned)
                avg_x2 = avg_x2/len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1

        # Step 5 :画出矩形框
        buff = 7
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff),int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff),int(rects[key][3]))
            cv2.rectangle(new_image,tup_topLeft,tup_botRight,(0,255,0),3)

        return new_image,rects

    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        if make_copy:
            new_image = np.copy(image)
        gap = 15.5   # 设置车位宽度
        spot_dict = {}  # 车位的坐标，一个车位对应一个位置
        tot_spots = 0
        # 对矩形框进行一些微调，使其覆盖面积更加准确
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}

        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}
        for key in rects:
            # 进行微调并画出矩形框
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])
            cv2.rectangle(new_image,(x1,y1),(x2,y2),(0,255,0),2)
            # 计算每一列可以分成多少车位
            num_splits = int(abs(y2-y1)//gap)
            # 画出每个车位
            for i in range(0,num_splits+1):
                y = int(y1 + i*gap)
                cv2.line(new_image,(x1,y),(x2,y),color,thickness)
            # 画出竖线，切分开每列的车位
            if key > 0 and key < len(rects)-1:
                x = int((x1+x2)/2)
                cv2.line(new_image,(x,y1),(x,y2),color,thickness)
            #计算车位数量
            if key == 0 or key == (len(rects)-1):  #如果是单排列，即第一排和最后一排
                tot_spots += num_splits +1
            else:#双排列
                tot_spots += 2*(num_splits+1)
            # 标记车位序号
            if key == 0 or key == (len(rects) - 1):
                for i in range(0,num_splits+1):
                    cur_len = len(spot_dict)
                    y =  int(y1+i*gap)
                    spot_dict[(x1,y,x2,y+gap)] =  cur_len + 1
            else:
                for i in range(0,num_splits+1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i*gap)
                    x = int((x1+x2)/2)
                    spot_dict[(x1,y, x, y + gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y + gap)] = cur_len + 2

        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename,new_image)

        return new_image,spot_dict

    def assign_spots_map(self, image, spot_dict, make_copy=True, color=[255, 0, 0], thickness=2):
        if make_copy:
            new_image = np.copy(image)
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
        return new_image

    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        for spot in spot_dict.keys():
            (x1,y1,x2,y2) = spot
            (x1,y1,x2,y2) = (int(x1),int(y1),int(x2),int(y2))
            # 裁剪该停车位区域
            spot_image = image[y1:y2,x1:x2]
            spot_image = cv2.resize(spot_image,(0,0),fx=2.0,fy=2.0)
            spot_id = spot_dict[spot]
            #保存
            filename = 'spot' + str(spot_id) + '.jpg'
            cv2.imwrite(os.path.join(folder_name,filename),spot_image)

    def make_prediction(self, image, model, class_dictionary):
        #预处理
        img = image/255.
        #转换成4D的tensor(用keras和tensorflow需要将3D图转化成4D)
        image = np.expand_dims(img,axis=0)

        #用模型训练
        class_predicted = model.predict(image)
        inID = np.argmax(class_predicted[0])
        label =  class_dictionary[inID]

        return label

    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0  #空车位个数
        all_spots = 0  #总个数
        for spot in spot_dict.keys():
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (48, 48))
            #对当前裁剪的车位进行识别
            label = self.make_prediction(spot_img,model,class_dictionary)
            if label == 'empty':
                cv2.rectangle(overlay,(x1,y1),(x2,y2),color,-1)
                cnt_empty += 1
        #合并原图和画出空车位的图
        cv2.addWeighted(overlay,alpha,new_image,1-alpha,0,new_image)

        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        save = False
        if save:
            filename = 'with_marking.jpg'
            cv2.imwrite(filename, new_image)
        self.cv_show('new_image', new_image)

        return new_image

    def predict_on_video(self, video_name, final_spot_dict, model, class_dictionary, ret=True):
        cap = cv2.VideoCapture(video_name)
        count = 0
        while ret:
            ret,image = cap.read()
            count += 1
            if count == 5:
                count = 0

                new_image = np.copy(image)
                overlay = np.copy(image)
                cnt_empty = 0
                all_spots = 0
                color = [0, 255, 0]
                alpha = 0.5

                for spot in final_spot_dict.keys():
                    all_spots += 1
                    (x1,y1,x2,y2) = spot
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))

                    spot_img = image[y1:y2,x1:x2]
                    spot_img = cv2.resize(spot_img,(48,48))

                    label = self.make_prediction(spot_img,model,class_dictionary)
                    if label == 'empty':
                        cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color, -1)
                        cnt_empty += 1

                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imshow('frame', new_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        cap.release()