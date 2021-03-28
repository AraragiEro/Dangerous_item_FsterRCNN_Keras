import keras.engine.topology as KE
import keras.layers as KL
import tensorflow as tf 
import numpy as np
from math import e

class proposal(KE.Layer):
    def __init__(self, nms_thresh, proposal_count, imagesize, **kwargs):
        super(proposal,self).__init__(**kwargs)
        self.imagesize = imagesize
        self.nms_thresh = nms_thresh
        self.proposal_count = proposal_count
        
    def call(self, inputs):
        #input[rpn2K,rpn4K,anchorsCWH]
        frontScore = inputs[0][:,:,1]
        idxs = tf.squeeze(tf.where(tf.greater(frontScore[0],0.7)),1)
        
        deltes = inputs[1][:]
        deltes = tf.gather(deltes[0],idxs)
        frontScore = tf.gather(frontScore[0],idxs)
        anchors = inputs[2][:]
        anchors = tf.gather(anchors[0],idxs)
        
        prenms_num = tf.minimum(2000,tf.shape(anchors)[0])
        idxs = tf.nn.top_k(frontScore,prenms_num).indices
        frontScore = tf.gather(frontScore,idxs)
        deltes = tf.gather(deltes,idxs)
        anchors = tf.gather(anchors,idxs)
        
        deltaX = deltes[:,0]
        deltaY = deltes[:,1]
        deltaW = deltes[:,2]
        deltaH = deltes[:,3]
        Pw = (e**deltaW)*anchors[:,2]
        Ph = (e**deltaH)*anchors[:,3]
        Px = (deltaX*anchors[:,2])+anchors[:,0]
        Py = (deltaX*anchors[:,3])+anchors[:,1]
        refined_anchorsCWH = tf.stack([Px, Py, Pw, Ph],1)
        
        center = refined_anchorsCWH[:,0:2]
        WH = refined_anchorsCWH[:,2:4]
        
        refined_anchorsSE = tf.concat([center-WH*0.5,center+WH*0.5],1)
        
        #NMS?
        idxs = tf.image.non_max_suppression(refined_anchorsSE,frontScore,25, 0.3)
        
        box = tf.cast(tf.gather(refined_anchorsSE, idxs),tf.int32)
        
        boxxs = tf.clip_by_value(box[:,0], 0,self.imagesize[1])
        boxxe = tf.clip_by_value(box[:,2], 0,self.imagesize[1])
        boxys = tf.clip_by_value(box[:,1], 0,self.imagesize[0])
        boxye = tf.clip_by_value(box[:,3], 0,self.imagesize[0])
        
        nms = tf.expand_dims(tf.stack([boxxs,boxys,boxxe,boxye],1),0)
        '''
        self.outputshape = tf.shape(nms)[0]
        '''
        return [nms,tf.expand_dims(self.imagesize,0)]
    
    #def compute_output_shape(self, input_shape):#？
        #return (inputs, 10)
    
    
class target_detector(KE.Layer):
    def __init__(self, imagesize, **kwargs):
        #[info['bbox']] x 2
        super(target_detector,self).__init__(**kwargs)
        self.imagesize = imagesize

        
        
    def IOU(self,bbox,anchors):
        xs1,ys1,xe1,ye1 = bbox[0],bbox[1],bbox[2],bbox[3]
        xs2,ys2,xe2,ye2 = anchors[:,0],anchors[:,1],anchors[:,2],anchors[:,3]
        xs = tf.maximum(xs1,xs2)
        xe = tf.minimum(xe1,xe2)
        ys = tf.maximum(ys1,ys2)
        ye = tf.minimum(ye1,ye2)
        xo = tf.maximum(xe-xs,0)
        yo = tf.maximum(ye-ys,0)
        overarea = xo*yo
        #xymul = (xs1-xe1)*(ys1-ye1)+(xs2-xe2)*(ys2-ye2)-overarea
        xymul = (xs1-xe1)*(ys1-ye1)#+(xs2-xe2)*(ys2-ye2)-overarea
        IOU = overarea / xymul
        return IOU
    
    def deltas(self, gt_bboxs, proposal, imgsize, classID):
        #gtbox N x 4 CWH
        #classID N
        #proposal N x 4 SE
        Ax = (proposal[:,2]+proposal[:,0])/2
        Ay = (proposal[:,3]+proposal[:,1])/2
        Aw = tf.abs(proposal[:,2]-proposal[:,0])
        Ah = tf.abs(proposal[:,3]-proposal[:,1])
        bboxs = tf.gather(gt_bboxs,classID)

        deltaX = (bboxs[:,0]-Ax)/Aw
        deltaY = (bboxs[:,1]-Ay)/Ah
        deltaW = tf.log(bboxs[:,2]/Aw)
        deltaH = tf.log(bboxs[:,3]/Ah)

        deltas = tf.stack([deltaX, deltaY, deltaW, deltaH],1)


        '''
        proposal = tf.expand_dims(proposal,1)
        proposal = tf.concat([proposal for i in range(len(gt_bboxs))],1)
        imgS = tf.concat([imgsize,imgsize],0)
        sizeDevidend = tf.stack([imgS for i in range(len(gt_bboxs))],1)
        deltes = -(proposal-gt_bboxs)/sizeDevidend
        deltes = tf.gather(deltes,classID)
        deltes /=0.01
        '''
        '''
        ps = proposal[:,0:2]
        pe = proposal[:,2:4]
        pc = (ps+pe)/2
        pwh = pe-ps
        proposalCWH = tf.concat([pc,pwh],1)
        shape = tf.shape(proposal)[0]
        proposal_ID = tf.cast(tf.expand_dims(tf.range(0,shape),-1),tf.int32)
        cla = tf.cast(tf.expand_dims(classID,-1),tf.int32)
        imgS = tf.stack([imgsize[1],imgsize[0]],0)
        k = tf.cast(tf.ones((tf.shape(proposal)[0],2)),tf.int32)
        imgDicidend = imgS * k 
        WHDividend = tf.cast(tf.gather(gt_bboxs,classID)[:,2:],tf.int32)
        dividend = tf.concat([imgDicidend,WHDividend],1)
        deltas = proposalCWH/tf.cast(dividend,tf.float32)
        deltas /= 0.01
        '''
        return deltas
    
    def call(self, inputs):
        #input[proposal,gt_bboxsCWH]
        proposal = tf.cast(inputs[0],tf.float32)
        gt_bboxsCWH1 = tf.concat([[[[1.0,1.0,1.0,1.0,0.0]]],inputs[1]],1)
        gt_bboxsCWH = gt_bboxsCWH1[:,:,:4]#gt N x 5 [x1,x2,y1,y2,classid]
        
        center = gt_bboxsCWH[:,:,0:2]
        WH = gt_bboxsCWH[:,:,2:4]
        gt_bboxsSE = tf.cast(tf.concat([center-WH/2,center+WH/2],2),tf.float32)
        gt_class = gt_bboxsCWH1[:,:,-1][0]
        IOU = tf.transpose(tf.map_fn(lambda x:self.IOU(x,proposal[0]),gt_bboxsSE[0],tf.float32))
        IOU = tf.slice(IOU,[0,1],[-1,-1])
        bk = tf.fill([tf.shape(IOU)[0],1],0.1)
        IOU = tf.concat([bk,IOU],1)
        classIdxs = tf.argmax(IOU, axis=1)
        classID = tf.gather(gt_class,classIdxs)
        
        deltas = self.deltas(gt_bboxsCWH[0], proposal[0],self.imagesize, classIdxs)
        ''''''
        #self.outputshape = tf.shape(proposal)[0]
        return [tf.expand_dims(classID,0), tf.expand_dims(deltas,0)]

class classifier(KE.Layer):
    def __init__(self, poolingsize, fp_size, filters, class_num, **kwargs):
        super(classifier,self).__init__(**kwargs)
        self.poolingsize = (poolingsize*2,poolingsize*2)
        self.fp_size = fp_size
        self.filters = filters
        self.class_num = class_num

    def roipooling(self, featuremap, proposal):
        x1 = proposal[0][:,0]
        y1 = proposal[0][:,1]
        x2 = proposal[0][:,2]
        y2 = proposal[0][:,3]
        proposal = tf.cast(tf.stack([y1/self.fp_size[0],x1/self.fp_size[1],y2/self.fp_size[0],x2/self.fp_size[1]],1),tf.float32)
        shape = tf.shape(proposal)[0]
        shape = tf.zeros(shape,dtype=tf.int32)
        
        featuremap = tf.image.crop_and_resize(featuremap,proposal,shape,self.poolingsize)
        ''''''
        
        return featuremap

    def call(self, inputs):
        #[featuremap, proposal]
        featuremap = inputs[0]
        proposal = inputs[1]
        filter = self.filters

        roipooling1 = self.roipooling(featuremap, proposal)
        roipooling = KL.pooling.MaxPool2D()(roipooling1)

        roipooling = KL.Conv2D(filter,(7,7),padding='valid',strides=2)(roipooling)       
        roipooling = KL.BatchNormalization(axis=3)(roipooling)
        roipooling = KL.Activation('relu')(roipooling)

        roipooling = KL.core.Flatten()(roipooling)
        
        roipooling = KL.Conv2D(4096,(1,1),padding='valid',strides=2)(roipooling)
        roipooling = KL.BatchNormalization(axis=3)(roipooling)
        roipooling = KL.Activation('relu')(roipooling)

        roipooling = KL.Conv2D(4096,(1,1),padding='valid',strides=2)(roipooling)
        roipooling = KL.BatchNormalization(axis=3)(roipooling)
        roipooling = KL.Activation('relu')(roipooling)
        
        Fclass = KL.Conv2D(self.class_num,(1,1))(roipooling)
        Fclass = KL.BatchNormalization(axis=3)(Fclass)
        Fclass = KL.Activation('softmax')(Fclass)
        Fclass = KL.Lambda(lambda x:tf.reshape(x,[1,-1,self.class_num]))(Fclass)

        Fdeltes = KL.Conv2D(4*self.class_num,(1,1))(roipooling)
        Fdeltes = KL.BatchNormalization(axis=3)(Fdeltes)
        Fdeltes = KL.Activation('linear')(Fdeltes)
        Fdeltes = KL.Lambda(lambda x:tf.reshape(x,[1,-1,self.class_num,4]))(Fdeltes)

        return [Fclass,Fdeltes]

        
