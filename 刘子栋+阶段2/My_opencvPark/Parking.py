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
        plt.tight_layout(pad=0,h_pad=0,w_pad=0)  # ����ʹx,y���labels�����ڸ�
        plt.show()

    def cv_show(self,name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def select_rgb_white_yellow(self,image):
        # ������С�������ֵ�����˵�����
        lower = np.uint8([120,120,120])
        upper = np.uint8([255,255,255])
        # cv2.inRange()���image(lower,upper)��Χ����Ĳ���ȫ����Ϊ0������֮��ı��255
        white_mask =  cv2.inRange(image,lower,upper)
        self.cv_show('white_mask',white_mask)
        # ������Ĥ��mask�����С��롱����������Ĥͼ���ɫ�����Ƕ���Ҫ����ͼ�����صı�����
        # ��ɫ�����Ƕ���Ҫ����ͼ�����ص��޳������ఴλ����ԭ������ֻ��Ч����ͬ����
        masked = cv2.bitwise_and(image,image,mask=white_mask)
        self.cv_show('masked',masked)
        return masked

    def convert_gray_scale(self, image):
        #�Ҷ�ͼ
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        #��Ե���
        return cv2.Canny(image,low_threshold,high_threshold)

    def filter_region(self, image, vertices):
        #ȥ������Ҫ�ĵط�
        mask = np.zeros_like(image)
        if len(mask.shape) == 2:
            # ���ݶ���λ�ã��Ѷ��㸲�ǵ��ڲ������������
            cv2.fillPoly(mask,vertices,255)
            self.cv_show('mask',mask)
        #����image��mask������,����������˵�
        return cv2.bitwise_and(image,mask)

    def select_region(self, image):
        #�ֶ�ѡ������
        #�����ú�Ҫѡ��������Ķ��㣬����Щ���������ӳ�һ������
        rows, cols = image.shape[:2]
        rows, cols = image.shape[:2]
        pt_1 = [cols * 0.05, rows * 0.90]  # ����1
        pt_2 = [cols * 0.05, rows * 0.70]  # ����_��2
        pt_3 = [cols * 0.30, rows * 0.55]  # ��3
        pt_4 = [cols * 0.6, rows * 0.15]   # ����4
        pt_5 = [cols * 0.90, rows * 0.15]  # ����5
        pt_6 = [cols * 0.90, rows * 0.90]  # ����6
        # ת��Ϊ3ά�ľ���
        vertices = np.array([[pt_1,pt_2,pt_3,pt_4,pt_5,pt_6]],dtype=np.int32)
        point_img = image.copy()
        #�Ҷ�ͼ��ԭΪ��ɫͼ
        point_img = cv2.cvtColor(point_img,cv2.COLOR_GRAY2BGR)
        #�Ӷ����л�ȡx,y���겢��������
        for p in vertices[0]:
            cv2.circle(point_img,(p[0],p[1]),10,(0,0,255),4)
        self.cv_show('point_img',point_img)
        
        return self.filter_region(image,vertices)

    def hough_lines(self, image):
        # �����ͼ���Ǳ�Ե����Ľ��
        # minLineLengh(�ߵ���̳��ȣ�������̵Ķ�������)��MaxLineCap������ֱ��֮����������С�ڴ�ֵ����Ϊ��һ��ֱ�ߣ�
        # rho���뾫��,theta�ǶȾ���,ԽСԽ��ȷ,threshod�����趨��ֵ�ű������߶�,ԽС����ֱ��Խ��
        return cv2.HoughLinesP(image,rho=0.1,theta=np.pi/10,threshold=15, minLineLength=9, maxLineGap=4)

    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        if make_copy:
            image = np.copy(image)
        # ���˻���任��⵽��ֱ��
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                # �����������������������ˮƽ�ģ��ߵĳ��������޵�
                if abs(y2-y1) <= 1 and abs(x2-x1) >= 25 and abs(x2-x1) <= 55:
                    cleaned.append((x1,y1,x2,y2))
                    cv2.line(image,(x1,y1),(x2,y2),color,thickness)
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)

        #Step 1:���˲���ֱ��
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        #Step 2:��ֱ�߰���x1,y1��������
        import operator
        list1 = sorted(cleaned,key=operator.itemgetter(0,1))

        #Step 3:�ҵ�����У��൱��ÿ����һ�ų�
        clusters = {}
        dIndex = 0
        clus_dist = 10 # �Ƿ�Ϊͬһ�е���ֵ
        for i in range(len(list1) - 1):
            distance = abs(list1[i+1][0] - list1[i][0])  # ����ֱ�ߵ�x1֮��ľ���
            #�������С����ֵ����Ϊͬһ��
            if distance <= clus_dist:
                if not dIndex in clusters.keys():
                    clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i+1])
            else:
                dIndex += 1

        #Step 4: �õ��е�����������������꣬������2�����껭�����ο�
        rects = {}
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 5:
                #��������ֱ�ߵ�x1,x2,y1,y2����������𣬵���𲻴�����ȡ��ƽ��ֵ��Ϊ�����
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

        # Step 5 :�������ο�
        buff = 7
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff),int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff),int(rects[key][3]))
            cv2.rectangle(new_image,tup_topLeft,tup_botRight,(0,255,0),3)

        return new_image,rects

    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        if make_copy:
            new_image = np.copy(image)
        gap = 15.5   # ���ó�λ���
        spot_dict = {}  # ��λ�����꣬һ����λ��Ӧһ��λ��
        tot_spots = 0
        # �Ծ��ο����һЩ΢����ʹ�串���������׼ȷ
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}

        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}
        for key in rects:
            # ����΢�����������ο�
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])
            cv2.rectangle(new_image,(x1,y1),(x2,y2),(0,255,0),2)
            # ����ÿһ�п��Էֳɶ��ٳ�λ
            num_splits = int(abs(y2-y1)//gap)
            # ����ÿ����λ
            for i in range(0,num_splits+1):
                y = int(y1 + i*gap)
                cv2.line(new_image,(x1,y),(x2,y),color,thickness)
            # �������ߣ��зֿ�ÿ�еĳ�λ
            if key > 0 and key < len(rects)-1:
                x = int((x1+x2)/2)
                cv2.line(new_image,(x,y1),(x,y2),color,thickness)
            #���㳵λ����
            if key == 0 or key == (len(rects)-1):  #����ǵ����У�����һ�ź����һ��
                tot_spots += num_splits +1
            else:#˫����
                tot_spots += 2*(num_splits+1)
            # ��ǳ�λ���
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
            # �ü���ͣ��λ����
            spot_image = image[y1:y2,x1:x2]
            spot_image = cv2.resize(spot_image,(0,0),fx=2.0,fy=2.0)
            spot_id = spot_dict[spot]
            #����
            filename = 'spot' + str(spot_id) + '.jpg'
            cv2.imwrite(os.path.join(folder_name,filename),spot_image)

    def make_prediction(self, image, model, class_dictionary):
        #Ԥ����
        img = image/255.
        #ת����4D��tensor(��keras��tensorflow��Ҫ��3Dͼת����4D)
        image = np.expand_dims(img,axis=0)

        #��ģ��ѵ��
        class_predicted = model.predict(image)
        inID = np.argmax(class_predicted[0])
        label =  class_dictionary[inID]

        return label

    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0  #�ճ�λ����
        all_spots = 0  #�ܸ���
        for spot in spot_dict.keys():
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (48, 48))
            #�Ե�ǰ�ü��ĳ�λ����ʶ��
            label = self.make_prediction(spot_img,model,class_dictionary)
            if label == 'empty':
                cv2.rectangle(overlay,(x1,y1),(x2,y2),color,-1)
                cnt_empty += 1
        #�ϲ�ԭͼ�ͻ����ճ�λ��ͼ
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