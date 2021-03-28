
# coding: utf-8

# In[1]:


from lxml import etree as ET
import numpy as np
import numpy as np
import tensorflow as tf
import keras.backend as K
from PIL import Image
import math
import cv2


dic = {
    '正常':0,
    '铁壳打火机':1,
    '黑钉打火机':2,
    '刀具':3,
    '电源和电池':4,
    '剪刀':5
}


# In[371]:


class dataset:
    labelDir = '/VOC2007/Annotations/'
    imageDir = '/VOC2007/JPEGImages/'
    fileListDir = '/VOC2007/ImageSets/Main/'
    def __init__(self,dir,type):
        self.labelDir = dir + self.labelDir
        self.imageDir = dir + self.imageDir
        self.type = type
        self.fileListDir = dir + self.fileListDir
        self.rpn_stride = 16
        self.__getTrainList()
        
        
        
    #def genBatch(self,batchSize):
        
    def XMLreader(self,fileName):
        dir = self.labelDir + fileName + '.xml'
        tree = ET.parse(dir)
        #root = tree.getroot()
        info = {}
        info['filename'] = tree.find('filename').text
        #info['size'] = [float(tree.find('size').find('width').text), float(tree.find('size').find('height').text)]
        bboxSE = []
        bboxCWH = []
        for ob in tree.findall('object'):
            size = [
                float(ob.find('bndbox').find('xmin').text)-3,
                float(ob.find('bndbox').find('ymin').text)-3,
                float(ob.find('bndbox').find('xmax').text)+3,
                float(ob.find('bndbox').find('ymax').text)+3
            ]
            bboxCWH.append(
                [
                    (size[0]+size[2])/2,
                    (size[1]+size[3])/2,
                    (size[2]-size[0]),
                    (size[3]-size[1]),
                    dic[ob.find('name').text]
                ])
            bboxSE.append(
                [
                    size[0],
                    size[1],
                    size[2],
                    size[3],
                    dic[ob.find('name').text]                    
                ])
        if bboxSE == []:
            info['bboxSE'] = bboxSE
        else:
            info['bboxSE'] = np.array(bboxSE).reshape(len(bboxSE),5)
        if bboxCWH == []:
            info['bboxCWH'] = bboxCWH
        else:
            info['bboxCWH'] = np.array(bboxCWH).reshape(len(bboxSE),5)
        self.info = info
        return tree
        #print(dir)
        
    def __getTrainList(self):
        ListDir = self.fileListDir + '/'+self.type+'.txt'
        reader = open(ListDir,mode='r')
        List = reader.read()
        List = List.split('\n')
        self.TrainList = List
        
    def anchor_gen(self,type = 'SE'):
        info = self.info
        size_X=self.size[0]
        size_Y=self.size[1]
        rpn_stride=self.rpn_stride
        a=32
        scales=[a,a*2,a*4]
        rations=[0.5,1,2]
        
        scales , rations = np.meshgrid( scales , rations )
        scales , rations = scales.flatten() , rations.flatten()
        scaleY = scales * np.sqrt(rations)
        scaleX = scales / np.sqrt(rations)
        #---------------------------------
        '''
        X * Y = scales^2
        X / Y = rations
        scales是面积关系，rations是X Y比值。
        解得如上关系。
        可以保证同种尺寸的面积一致。
        '''
        #--------------------------------

        shiftX = np.arange(0,math.ceil(size_X/rpn_stride)) * rpn_stride
        shiftY = np.arange(0,math.ceil(size_Y/rpn_stride)) * rpn_stride
        shiftX,shiftY = np.meshgrid(shiftX,shiftY)
        centerX,anchorX = np.meshgrid(shiftX,scaleX)
        centerY,anchorY = np.meshgrid(shiftY,scaleY)
        anchor_center = np.stack([centerX,centerY],axis = 2).reshape(-1,2)
        anchor_size = np.stack([anchorX,anchorY],axis = 2).reshape(-1,2)
        boxes1 = np.concatenate([anchor_center - 0.5*anchor_size,anchor_center + 0.5*anchor_size],axis = 1)
        boxes2 = np.concatenate([anchor_center,anchor_size],axis = 1)
        self.anchorsSE = boxes1
        self.anchorsCWH = boxes2
        #self.info = info
        if type == 'SE':
            return boxes1
        elif type == 'CWH':
            return boxes2

    
    def IOU(self,bboxs,anchors):
        IOU = []
        for bbox in bboxs:
            xs1,ys1,xe1,ye1 = bbox[0,0],bbox[0,1],bbox[1,0],bbox[1,1]
            xs2,ys2,xe2,ye2 = anchors[:,0,0],anchors[:,0,1],anchors[:,1,0],anchors[:,1,1]
            xs = np.maximum(xs1,xs2)
            xe = np.minimum(xe1,xe2)
            ys = np.maximum(ys1,ys2)
            ye = np.minimum(ye1,ye2)
            xo = np.maximum(xe-xs,0)
            yo = np.maximum(ye-ys,0)
            '''
            bS = bbox[0]
            bE = bbox[1]
            aS = np.array(anchors[:,0])
            aE = np.array(anchors[:,1])
            
            xs1,ys1,xe1,ye1 = bS[0],bS[1],bE[0],bE[1]
            xs2,ys2,xe2,ye2 = aS.T[0],aS.T[1],aE.T[0],aE.T[1]
            c1,c2,c3,c4 = xs1-xs2, xs1-xe2, xe1-xs2, xe1-xe2
            q1 = ((c1<0) * (c2<0) * (c3>0) * (c4<0)) +((c1>0) * (c2<0) * (c3>0) * (c4>0))
            q2 = ((c1<0) * (c2<0) * (c3>0) * (c4>0)) +((c1>0) * (c2<0) * (c3>0) * (c4<0))
            q3 = ((c1<0) * (c2<0) * (c3<0) * (c4<0)) +((c1>0) * (c2>0) * (c3>0) * (c4>0))
            xo1 = q1*(np.greater((xe1-xs2)-(xe2-xs1),0)*(xe2-xs1)+np.greater((xe2-xs1)-(xe1-xs2),0)*(xe1-xs2))+\
                q2*(np.less((xe1-xs1)-(xe2-xs2),0)*(xe1-xs1)+np.greater((xe1-xs1)-(xe2-xs2),0)*(xe2-xs2))

            c1,c2,c3,c4 = ys1-ys2, ys1-ye2, ye1-ys2, ye1-ye2
            q1 = ((c1<0) * (c2<0) * (c3>0) * (c4<0)) +((c1>0) * (c2<0) * (c3>0) * (c4>0))
            q2 = ((c1<0) * (c2<0) * (c3>0) * (c4>0)) +((c1>0) * (c2<0) * (c3>0) * (c4<0))
            q3 = ((c1<0) * (c2<0) * (c3<0) * (c4<0)) +((c1>0) * (c2>0) * (c3>0) * (c4>0))
            yo1 = q1*(np.greater((ye1-ys2)-(ye2-ys1),0)*(ye2-ys1)+np.greater((ye2-ys1)-(ye1-ys2),0)*(ye1-ys2))+\
                q2*(np.less((ye1-ys1)-(ye2-ys2),0)*(ye1-ys1)+np.greater((ye1-ys1)-(ye2-ys2),0)*(ye2-ys2))
            '''
            overarea = xo*yo
            xymul = (xs1-xe1)*(ys1-ye1)+(xs2-xe2)*(ys2-ye2)-overarea
            xymul = overarea / xymul
            IOU.append(xymul)
        return np.array(IOU)
    
    def rpn_bbox(self,bboxs,anchors,input_rpn_match,IOU):

        idxs = np.argmax(IOU,axis=1)
        bboxs = bboxs[idxs]

        deltaX = (bboxs[:,0]-anchors[:,0])/anchors[:,2]
        deltaY = (bboxs[:,1]-anchors[:,1])/anchors[:,3]
        deltaW = np.log(bboxs[:,2]/anchors[:,2])
        deltaH = np.log(bboxs[:,3]/anchors[:,3])
        deltas = np.stack([deltaX,deltaY,deltaW,deltaH],axis=1) 

        #size = self.size
        #sizeDevidend = np.array([size,size]).reshape(4)
        #deltes = -(anchors-bboxs)/sizeDevidend
        '''
        anchors = np.expand_dims(anchors,axis=1)
        anchors = np.concatenate([anchors for i in range(len(bboxs))],axis=1)
        size = self.size
        sizeDevidend = np.array([size,size]).reshape(4)
        sizeDevidend = np.array([sizeDevidend for i in range(len(bboxs))])
        deltes = -(anchors-bboxs)/sizeDevidend
        
        num = np.array(range(len(idxs)))
        idxs = np.stack([num,idxs],axis=1)
        deltes = tf.gather_nd(deltes,idxs)
        '''
        deltas = deltas 
        return deltas
        
    def rpnInputData(self,filename):
        bboxsSE = np.array(self.bboxSE[:,:4]).reshape(-1,2,2)
        IOU = self.IOU(bboxsSE,self.anchorsSE.reshape(-1,2,2))
        input_rpn_match = np.sort(IOU.T,axis=1).T[-1]
        input_rpn_match = np.greater_equal(input_rpn_match,0.5)*1 + np.less_equal(input_rpn_match,0.3)*(-1)

        bboxsCWH = np.array(self.bboxCWH[:,:4]).reshape(-1,2,2)        
        rpn_bbox = self.rpn_bbox(bboxsCWH.reshape(-1,4),self.anchorsCWH,input_rpn_match,IOU.T)
        return input_rpn_match.reshape(-1,1),rpn_bbox

        
    def inputImgData(self,filename):
        self.XMLreader(filename)
        dir = self.imageDir + filename + '.jpg'
        img = np.array(cv2.imread(dir))
        x = img.shape[1]
        y = img.shape[0]
        k = 1
        if x > 800:
            img = cv2.resize(img,(800,int(y*800/x)))
            k=k*800/x
        y = img.shape[0]
        x = img.shape[1]
        if y > 480:
            img = cv2.resize(img,(int(x*480/y),480))
            k=k*480/y
        size=[img.shape[1],img.shape[0]]
        #info = self.info
        #info['size']=size
        self.size = size
        '''
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        img = cv2.merge((bH, gH, rH))
        '''
        bboxSE = self.info['bboxSE']
        bboxSE_ = bboxSE[:,:4]*k
        c = np.expand_dims(bboxSE[:,-1],1)
        bboxSE = np.concatenate([bboxSE_,c],1)
        self.bboxSE = np.array(bboxSE).astype('int32')

        bboxCWH = self.info['bboxCWH']
        bboxCWH_ = bboxCWH[:,:4]*k
        c = np.expand_dims(bboxCWH[:,-1],1)
        bboxCWH = np.concatenate([bboxCWH_,c],1)
        self.bboxCWH = np.array(bboxCWH).astype('int32')
        return img

    def gen_batch(self):
        TL = self.TrainList[:-2]
        while 1: 
            np.random.shuffle(TL)
            self.XMLreader(TL[0])
            if self.info['bboxSE'] != []:
                img = self.inputImgData(TL[0])
                self.anchor_gen()
                input_rpn_match,rpn_bbox = self.rpnInputData(TL[0])
                input_rpn_match = np.array(input_rpn_match)
                yield [ np.expand_dims(img,axis=0),
                        np.expand_dims(self.bboxCWH,axis=0),
                        np.expand_dims(self.anchorsCWH,axis=0),
                        np.expand_dims(input_rpn_match,axis=0),
                        np.expand_dims(rpn_bbox,axis=0)
                        ],[]
                        



