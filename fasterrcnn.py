import keras.layers as KL
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import keras
import numpy as np
from layers import target_detector, proposal, classifier
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import os


class faster_rcnn():
    def __init__(self):
        self.model_save_dir='./modelWeight'

    def __building_block(self, filters, block, stage, trainable = False):
        conv_name_base = 'res' + str(block) + stage + '_branch'
        bn_name_base = 'bn' + str(block) + stage + '_branch'
        if stage !='a':
            stride = 1
        else:
            stride = 2
        def f(x):
            y = KL.Conv2D(filters,(1,1),strides=stride,trainable=trainable,name=conv_name_base+'2a',bias_initializer='glorot_uniform')(x)
            y = KL.BatchNormalization(axis=3,name=bn_name_base+'2a')(y)
            y = KL.Activation('relu')(y)
            
            y = KL.Conv2D(filters,(3,3),trainable=trainable,padding='same',name=conv_name_base+'2b',bias_initializer='glorot_uniform')(y)
            y = KL.BatchNormalization(axis=3,name=bn_name_base+'2b')(y)
            y = KL.Activation('relu')(y)        
        
            y = KL.Conv2D(4*filters,(1,1),trainable=trainable,name=conv_name_base+'2c',bias_initializer='glorot_uniform')(y)
            y = KL.BatchNormalization(axis=3,name=bn_name_base+'2c')(y)
            
            if stage == 'a':
                shortcut = KL.Conv2D(4*filters,(1,1),trainable=trainable,strides=stride,bias_initializer='glorot_uniform')(x)
                shortcut = KL.BatchNormalization(axis=3)(shortcut)
            else:
                shortcut = x
            y = KL.Add()([y,shortcut])
            y = KL.Activation('relu')(y)
            return y
        return f

    def __resnet_featureExtractor(self, inputs, filters, blocks, f_trainable=False):
        filters = filters
        x = KL.BatchNormalization(axis=3)(inputs)
        x = KL.Conv2D(filters,(7,7),trainable=f_trainable,padding='same',name = 'conv1',bias_initializer='glorot_uniform')(x)
        x = KL.BatchNormalization(axis=3,name='bn_conv1')(x)
        x = KL.Activation('relu')(x)
        
        
        blocks = blocks
        #--stage start--#
        #--stage2--#
        x = self.__building_block(filters, block=2, stage='a',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=2, stage='b',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=2, stage='c',trainable=f_trainable)(x)
        filters *= 2

        #--stage3--#
        x = self.__building_block(filters, block=3, stage='a',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=3, stage='b',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=3, stage='c',trainable=f_trainable)(x)        
        x = self.__building_block(filters, block=3, stage='d',trainable=f_trainable)(x)
        filters *= 2   
        
        #--stage4--#
        x = self.__building_block(filters, block=4, stage='a',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=4, stage='b',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=4, stage='c',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=4, stage='d',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=4, stage='e',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=4, stage='f',trainable=f_trainable)(x)
        filters *= 2
        
        #--stage4--#
        x = self.__building_block(filters, block=5, stage='a',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=5, stage='b',trainable=f_trainable)(x)
        x = self.__building_block(filters, block=5, stage='c',trainable=f_trainable)(x)
        filters *= 2

        '''
        for i,blocknum in enumerate(blocks):
            for block_id in range(blocknum):
                x = self.__building_block(filters, block_id, i+2)(x)
            filters *= 2
        '''
        return x

    def __rpn_net(self, inputs, k):
        shared_map = KL.Conv2D(1024,(3,3),padding='same',bias_initializer='glorot_uniform')(inputs)
        shared_map = KL.BatchNormalization(axis=3)(shared_map)
        shared_map = KL.Activation('relu')(shared_map)
        rpn_class = KL.Conv2D(1024,(1,1),bias_initializer='glorot_uniform')(shared_map)
        rpn_class = KL.BatchNormalization(axis=3)(rpn_class)
        rpn_class = KL.Activation('linear')(rpn_class)
        rpn_class = KL.Conv2D(2*k,(1,1),bias_initializer='glorot_uniform')(shared_map)
        rpn_class = KL.BatchNormalization(axis=3)(rpn_class)       
        rpn_class = KL.Lambda(lambda x:tf.reshape(x,[tf.shape(x)[0],-1,2]))(rpn_class)
        rpn_prob = KL.Activation('softmax')(rpn_class)
        
        y = KL.Conv2D(1024,(1,1),bias_initializer='glorot_uniform')(shared_map)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('relu')(y)
        y = KL.Conv2D(4*k,(1,1),bias_initializer='glorot_uniform')(y)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('linear')(y)
        rpn_bbox = KL.Lambda(lambda x:tf.reshape(x,[tf.shape(x)[0],-1,4]))(y)

        return rpn_class, rpn_prob, rpn_bbox

    def __roipooling(self, featuremap, proposal, fp_size, poolingsize):
        stride = tf.constant([16])
        x1 = proposal[0][:,0]/stride
        y1 = proposal[0][:,1]/stride
        x2 = proposal[0][:,2]/stride
        y2 = proposal[0][:,3]/stride
        fp_size = tf.cast(fp_size,tf.float64)
        proposal = K.cast(K.stack([y1/fp_size[0],x1/fp_size[1],y2/fp_size[0],x2/fp_size[1]],1),tf.float32)
        shape = K.shape(proposal)[0]
        shape = tf.zeros(shape,dtype=tf.int32)
        
        featuremap = KL.Lambda(lambda x:tf.image.crop_and_resize(featuremap,proposal,shape,poolingsize))([])
        ''''''
        
        return featuremap

    def __classifier(self, poolingsize, fp_size, filters, class_num,featuremap, proposal):
        #[featuremap, proposal]
        filter = filters
        poolingsize = tf.constant([poolingsize*2,poolingsize*2])

        roipooling1 = self.__roipooling(featuremap, proposal, fp_size, poolingsize)
        roipooling = KL.pooling.MaxPool2D()(roipooling1)
        '''
        roipooling = KL.Conv2D(filter,(7,7),padding='valid',strides=2)(roipooling)       
        roipooling = KL.BatchNormalization(axis=3)(roipooling)
        roipooling = KL.Activation('relu')(roipooling)
        '''
        roipooling = KL.Conv2D(filter,(3,3),padding='valid',strides=2,bias_initializer='glorot_uniform')(roipooling)       
        roipooling = KL.BatchNormalization(axis=3)(roipooling)
        roipooling = KL.Activation('relu')(roipooling)

        roipooling = KL.Conv2D(filter,(3,3),padding='valid',strides=2,bias_initializer='glorot_uniform')(roipooling)       
        roipooling = KL.BatchNormalization(axis=3)(roipooling)
        roipooling = KL.Activation('relu')(roipooling)

        Fclass = KL.Conv2D(4096,(1,1),padding='valid',strides=2,bias_initializer='glorot_uniform')(roipooling)
        Fclass = KL.BatchNormalization(axis=3)(Fclass)
        Fclass = KL.Activation('relu')(Fclass)
        Fclass = KL.Conv2D(4096,(1,1),padding='valid',strides=2,bias_initializer='glorot_uniform')(Fclass)
        Fclass = KL.BatchNormalization(axis=3)(Fclass)
        Fclass = KL.Activation('linear')(Fclass)
        Fclass = KL.Conv2D(class_num,(1,1),bias_initializer='glorot_uniform')(Fclass)
        Fclass = KL.BatchNormalization(axis=3)(Fclass)
        Fclass = KL.Activation('softmax')(Fclass)
        Fclass = KL.Lambda(lambda x:tf.reshape(x,[1,-1,class_num]))(Fclass)


        Fdeltes = KL.Conv2D(4096,(1,1),padding='valid',strides=2,bias_initializer='glorot_uniform')(roipooling)
        Fdeltes = KL.BatchNormalization(axis=3)(Fdeltes)
        Fdeltes = KL.Activation('relu')(Fdeltes)
        Fdeltes = KL.Conv2D(4*class_num,(1,1),bias_initializer='glorot_uniform')(Fdeltes)
        Fdeltes = KL.BatchNormalization(axis=3)(Fdeltes)
        Fdeltes = KL.Activation('linear')(Fdeltes)
        Fdeltes = KL.Lambda(lambda x:tf.reshape(x,[1,-1,class_num,4]))(Fdeltes)

        return Fclass,Fdeltes

    def __rpn_class_loss(self, rpn_match, rpn_class_logits):
        ## rpn_match (None, 576, 1)
        ## rpn_class_logits (None, 576, 2)
        '''
        rpn_match = tf.squeeze(rpn_match, -1)
        rpn_match = tf.cast(rpn_match,tf.float32)

        idxs_matchNotZero = K.cast(K.not_equal(rpn_match,0),tf.float32)
        rpn_class_logits = rpn_class_logits*idxs_matchNotZero+K.cast(K.equal(rpn_match,0),tf.float32)
        idxs_match_Not_Minusone = K.cast(K.not_equal(rpn_match,-1),tf.float32)
        idxs_match_Minusone = K.cast(K.equal(rpn_match,-1),tf.float32)
        shape_match_Minusone = K.sum(idxs_match_Minusone)
        shape_match_One = K.sum(K.cast(K.equal(rpn_match,1),tf.float32))
        mnusOne_need = K.cast(100-shape_match_One,tf.int32)
        
        idxs1 = idxs_match_Minusone/2
        idxs2 = tf.where(K.equal(rpn_match,-1))
        where = tf.random_uniform([mnusOne_need],minval=0,maxval=K.cast(shape_match_Minusone,tf.int32),dtype=tf.int32)
        #idxs2 = idxs2[where]

        #rpn_match = rpn_match*idxs_matchMinusoneToZero


        '''
        rpn_match = tf.squeeze(rpn_match, -1)
        indices_one = tf.where(K.equal(rpn_match,1))
        minusone_num = 100-K.sum(K.cast(K.equal(rpn_match,1),tf.int32))
        minusone_num = K.switch(minusone_num>100,minusone_num,100)
        shape = tf.shape(rpn_match)[1]
        minusone_num1 = tf.random_uniform([minusone_num],minval=0,maxval=minusone_num,dtype=tf.int32)
        indices_minusone = tf.gather(tf.where(K.equal(rpn_match,-1)),minusone_num1)
        indices = tf.concat([indices_one,indices_minusone],0)
        
        rpn_class_logits = tf.gather_nd(rpn_class_logits,indices) 
        anchor_class = tf.gather_nd(rpn_match, indices)

        minus_one = K.equal(anchor_class, 1)
        minus_one = K.cast(minus_one, tf.int32)   
        anchor_class = tf.cast(anchor_class,tf.int32) *minus_one
        concat = tf.range(start=0,limit=tf.shape(anchor_class)[0],delta=1)
        anchor_class_idxs = tf.stack([concat,anchor_class],axis=1)
        rpn_class_logits = tf.gather_nd(rpn_class_logits,anchor_class_idxs)
        loss = -K.log(rpn_class_logits)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        
        
        
         ###prediction
        ###target
        #loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=K.argmax(rpn_class_logits),labels=anchor_class)
        #loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        
        return loss#K.expand_dims(rpn_class_logits,0)#

    def __rpn_bbox_loss(self, target_bbox, rpn_match, rpn_bbox, lossthresh=0.5):
        

        rpn_match = tf.squeeze(rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)
        batch_counts = K.equal(rpn_match, 1)
        batch_counts = K.cast(batch_counts, tf.int32)
        batch_counts = K.sum(batch_counts, axis=1)
        target_bbox = tf.gather_nd(target_bbox, indices)
        diff = K.abs(target_bbox - rpn_bbox)
        less_than_one = K.cast(K.less(diff, 1.0), 'float32')
        loss = (less_than_one * lossthresh * diff**2) + ((1 - less_than_one) * lossthresh * diff)
        loss = K.switch(tf.size(loss) > 0,K.mean(loss), tf.constant(0.0))
        return loss
        
    def __classifier_deltes_loss(self, target_deltes, Fdeltes, classID, lossthresh=0.5):
        idxs = tf.where(tf.not_equal(classID,0.0))
        classID = tf.gather_nd(classID,idxs)
        t = tf.gather_nd(target_deltes,idxs)
        F = tf.gather_nd(Fdeltes,idxs)
        
        shape = tf.range(start=0,limit=tf.shape(classID)[0],delta=1,dtype=tf.int32)
        idxs = tf.stack([shape,K.cast(K.reshape(classID,[-1]),tf.int32)],axis=1)
        
        F = tf.gather_nd(F,idxs)
        
        diff = K.abs(t-F)
        
        lessTone = K.cast(K.less(diff, 1.0), "float32")
        loss = (lessTone *lossthresh* diff**2) + (1-lessTone)*diff
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        ''''''
        return loss
        
    def __classifier_class_loss(self, ClassID,Fclass):
        ClassID = K.cast(ClassID,tf.int32)
        notzeroidxs = tf.where(tf.not_equal(ClassID[0],0)) 
        zeroidxs = tf.where(tf.equal(ClassID[0],0))
        s = K.switch(tf.shape(notzeroidxs)[0]/6<1, 1, tf.cast(tf.shape(notzeroidxs)[0]/6,tf.int32))
        idxs0 = tf.concat([notzeroidxs,zeroidxs[:s,:]],axis=0)
        ClassID = tf.gather(ClassID[0],idxs0)
        Fclass = tf.gather(Fclass[0],idxs0)


        shape = tf.range(start=0,limit=tf.shape(ClassID)[0],delta=1,dtype=tf.int32)
        
        idxs = tf.stack([shape,K.reshape(ClassID,[-1])],axis=1)
        
        logits = tf.gather_nd(K.reshape(Fclass,[-1,6]),idxs)
        loss = -K.log(logits)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        ''''''
        return loss

    def build_model(self, modeltype='all', lr=0.0005, momentum=0.9,f_trainable=False):
        input_image = KL.Input(shape=[None,None,3], dtype=tf.float32)
        input_gtCWH = KL.Input(shape=[None,5], dtype=tf.float32)
        input_anchorsCWH = KL.Input(shape=[None,4], dtype=tf.float32)
        input_rpn_match = KL.Input(shape=[None,1], dtype=tf.int32)
        input_rpn_bbox = KL.Input(shape=[None,4], dtype=tf.float32)

        feature_map = self.__resnet_featureExtractor(input_image, 64, [3,4,6,3],f_trainable=f_trainable)
        rpn_class, rpn_prob, rpn_bbox = self.__rpn_net(feature_map, 9)
        proposals,imgsieze = proposal(0.8, 300, tf.shape(input_image)[1:3])([rpn_prob,rpn_bbox,input_anchorsCWH])#input[rpn2K,rpn4K,anchorsCWH]   
        if modeltype!='rpn':
            classID, target_deltes = target_detector(tf.shape(input_image)[1:3])([proposals,input_gtCWH])#input[proposal,imgsize,gt_bboxsCWH]
            Fclass, Fdeltes = self.__classifier(7,tf.shape(feature_map)[1:3],512,6,feature_map,proposals)
        
        loss_rpn_match = KL.Lambda(lambda x:self.__rpn_class_loss(*x), name='loss_rpn_match')([input_rpn_match, rpn_prob])
        loss_rpn_bbox = KL.Lambda(lambda x:self.__rpn_bbox_loss(*x), name='loss_rpn_bbox')([input_rpn_bbox, input_rpn_match, rpn_bbox])
        if modeltype!='rpn':
            loss_classifier_deltes = KL.Lambda(lambda x:self.__classifier_deltes_loss(*x), name='loss_classifier_deltes')([target_deltes,Fdeltes,classID])
            loss_classifier_class = KL.Lambda(lambda x:self.__classifier_class_loss(*x), name='loss_classifier_class')([classID,Fclass])

        
        if modeltype=='rpn':
            model = Model([input_image,input_gtCWH,input_anchorsCWH,input_rpn_match,input_rpn_bbox],
                [loss_rpn_match,loss_rpn_bbox,proposals,imgsieze])
        elif modeltype=='all':
            model = Model([input_image,input_gtCWH,input_anchorsCWH,input_rpn_match,input_rpn_bbox],
                [loss_rpn_match,loss_rpn_bbox,loss_classifier_deltes,loss_classifier_class])
        elif modeltype=='test':
            model = Model([input_image,input_gtCWH,input_anchorsCWH,input_rpn_match,input_rpn_bbox],
                [loss_rpn_match,loss_rpn_bbox,loss_classifier_deltes,loss_classifier_class,Fclass,Fdeltes,proposals,imgsieze,classID,Fclass])

        if modeltype=='rpn':
            model.load_weights('resnet50.h5',by_name=True)
        elif modeltype=='all':
            model.load_weights('resnet50.h5',by_name=True)
        ''''''
        rlc = model.get_layer('loss_rpn_match').output
        rld = model.get_layer('loss_rpn_bbox').output
        model.add_loss(rlc)
        model.add_loss(rld)
        if modeltype=='all':
            cfc = model.get_layer('loss_classifier_class').output
            cfd = model.get_layer('loss_classifier_deltes').output
            model.add_loss(cfc)
            model.add_loss(cfd)
        
        model.compile(loss=[None]*len(model.output), optimizer=keras.optimizers.Adam(lr=lr),
                            metrics=['accuracy'])
        
        model.metrics_names.append('r_c')
        model.metrics_tensors.append(rlc)
        model.metrics_names.append('r_delte')
        model.metrics_tensors.append(rld)
        if modeltype=='all':
            model.metrics_names.append('c_c')
            model.metrics_tensors.append(cfc)
            model.metrics_names.append('c_delte')
            model.metrics_tensors.append(cfd)                
        self.model = model

    def train(self, Tdata_generater, Vdata_generater, epoch_size, epoch, initial_epoch=1):
        model_name = 'keras_cifar10_trained_model.h5'
        filepath = "model_{epoch:02d}-{loss:.4f}.hdf5"

        checkpoint = ModelCheckpoint(os.path.join(self.model_save_dir,filepath),
                                        monitor='loss',
                                        verbose=1,
                                        save_weights_only=True,
                                        save_best_only=True)
        ReducelLR = ReduceLROnPlateau(monitor='loss',
                                        factor=0.5,
                                        patience=10,verbose=1,
                                        )

        if not os.path.isdir(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        history = self.model.fit_generator(Tdata_generater,steps_per_epoch=epoch_size,
                                                epochs=epoch, callbacks=[checkpoint,ReducelLR],
                                                initial_epoch=initial_epoch)

        model_path = os.path.join(self.model_save_dir, model_name)
        self.model.save_weights(model_path)
        print('Saved trained model at %s ' % model_path)   
        return history     
