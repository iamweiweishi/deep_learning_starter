############################################################################################    
# -*- coding: utf-8 -*- 
#!/usr/bin/python2.7  
#Author  : weiweishi    
#Date    : 12.08.2016   
# When the data set is prepared as stated in 'README.md', the next step is to produce a *.csv file,
# which is a standard file formant in TensorFlow. We neglect the transfer process here.
# Functions:  
# 1. read training_index.txt (the size of images is 28*28)
# 2. preprocess the images in the data set
# 3. training deep nn net
# 4. save the prediction results
##########################################################################################  
  
  
import tensorflow as tf  
import numpy as np  
import math  
import cv2  
import sys  
import os  
import os.path  
from scipy import ndimage  



#-------read the training_index.txt--------#  
  
'''
training_index = './training_index.txt'  
modelpath="./tmp/"  
classnum=36  
total_image = 6297*3  
global_step = 0 
max_epoch = 5000
'''

######Global Para#
imageSize=28
classnum=5
training_index = './csv.txt'  
modelpath="~/myexperiments/tmp/" 
max_epoch = 3000
global_step = 0 
total_image = 2500


global_idx = np.arange(total_image)  
images_g = np.array((total_image, 784))  
labels_g = np.array((total_image,classnum))  
  
def read_traing_list():  
    train_image_dir = []  
    train_label_dir = []  
    reader = open(training_index)  
    while 1:  
        line = reader.readline()  
                #print line  
        tmp = line.split(" ")  
        # print tmp  
        if not line:  
            break  
        train_image_dir.append(tmp[0])  
        train_label_dir.append(tmp[1][0:-1])  
    #print train_image_dir[1:total_image]  
    #print train_label_dir[1:total_image]  
    return train_image_dir, train_label_dir  


#------run the preprocess-------#  
  
class Dadaset(object):  
    def __init__(self, image, label,dtype=tf.float32):  
        self._image = image  
        self.label = label  
    @property  
    def image(self):  
        return self._image  
    @property  
    def label(self):  
        return self._label  
  
#------shift operation--------  
  
def getBestShift(img):  
    cy,cx = ndimage.measurements.center_of_mass(img)  
    rows,cols = img.shape  
    shiftx = np.round(cols/2.0-cx).astype(int)  
    shifty = np.round(rows/2.0-cy).astype(int)  
    return shiftx, shifty  
  
def shift(img,shiftx,shifty):  
    rows, cols = img.shape  
    M = np.float32([[1,0,shiftx],[0,1,shifty]])  
    shifted = cv2.warpAffine(img,M,(cols,rows))  
    return shifted    
  
def read_image():
    '''
    #create qu
    filename_queue = tf.train.string_input_producer(["train.tfrecords"])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #return
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [imageSize, imageSize, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    
    print type(img)
      
    global images_g  
    global labels_g 

    images_g = img 
    labels_g = label

    return img, label
    '''
    train_image_dir, train_label_dir = read_traing_list()  
    images = []  
    labels = []  
    ss=[]  
    index = 0  
    global images_g  
    global labels_g  
    #print len(train_image_dir)  
    #print('$$$$$: %s',train_image_dir[0])
    for idx in range(len(train_image_dir)):  
        image_tmp = cv2.imread(str(train_image_dir[idx]),cv2.IMREAD_GRAYSCALE)         
        shiftx, shifty = getBestShift(image_tmp)  
        gray = shift(image_tmp,shiftx,shifty)  
        print train_image_dir[idx],train_label_dir[idx]  
        #cv2.imshow("1",image_tmp)  
        #cv2.waitKey(12)  
        # print "jjgjg",image_tmp.shape  
        label_tmp = int(str(train_label_dir[idx]))  
        label_tmp2 = [0]*classnum  
        label_tmp2[label_tmp] = 1  
        images.append(image_tmp.flatten()/255.0)  
        # print type(image_tmp), image_tmp.shape
        # print type(images),len(images)
        # print label_tmp  
        # print "label:", label_tmp2  
        # print "label len:", len(labels)  
        labels.append(label_tmp2)  
        ss.append(train_image_dir[idx])  
        index += 1    
        # print ss[100],labels[100]  
    images = np.array(images)  
    labels = np.array(labels)  
    images_g = images  
    labels_g = labels  
    return images, labels  
# read_image()  
  
  
  
def next_batch(batch_size):  
    global global_step  
    global labels_g  
    global images_g  
    print global_step  
    global global_idx  
    # print labels_g.shape  
    start = batch_size * global_step  
    end = batch_size * (global_step+1)  
    global_step += 1  
    if(end >= total_image):  
        np.random.shuffle(global_idx)  
        global_step = 0  
        start = 0  
        end = batch_size  
        images_g[:] = images_g[global_idx]  
        labels_g[:] = labels_g[global_idx]  
        print images_g.shape  
    return images_g[start:end], labels_g[start:end]  
  
def generate_test_image(test_size):  
  
    global images_g  
    global labels_g  
    global global_idx  
    np.random.shuffle(global_idx)  
    images_g[:] = images_g[global_idx]  
    labels_g[:] = labels_g[global_idx]  
    return images_g[0:test_size], labels_g[0:test_size]  
  
#--------run CNN model-----#  
  
def CNNmodel():  
    x = tf.placeholder(tf.float32,[None,28*28])  
    y_ = tf.placeholder(tf.float32,[None,classnum])  
    def weight_variable(shape):  
        init = tf.truncated_normal(shape,stddev = 0.1)  
        return tf.Variable(init)  
    def bias_variable(shape):  
        init = tf.constant(0.1, shape = shape)  
        return tf.Variable(init)  
  
    ## convolution and pooling  
    ## convolution step=1，padding=0
    ## pooling : 2X2 max pool  
    def conv2d(x,W):  
        # strides:[batch, in_height, in_width, in_channels]  
        return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')  
  
    def maxpool2d(x):  
        return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')  
  
    ## model construction
    # patch_size :5*5; output:32 feature maps  
    # [5,5,1,32]
    x_image = tf.reshape(x,[-1,28,28,1])  
  
    W_conv1 = weight_variable([5,5,1,32])  
    b_conv1 = bias_variable([32])  
  
    #conv, relu, maxpool  
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)  
    h_pool1 = maxpool2d(h_conv1)  
  
    # the second layer
    W_conv2 = weight_variable([5,5,32,64])  
    b_conv2 = bias_variable([64])  
  
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)  
    h_pool2 = maxpool2d(h_conv2)  
  
    # fully connected layer:1024*1,
    # the image size: 7X7 (*64)
  
    W_fc1 = weight_variable([7*7*64,1024])  
    b_fc1 = bias_variable([1024])  
  
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])  
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)  
  
    # dropout step is incorprated during training;(no dropout during testing)
    # 'keep_prob = 1' means no dropout  
    keep_prob = tf.placeholder("float") #keep_prob ratio 
  
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)  
  
    #add softmax layer
  
    W_fc2 = weight_variable([1024,classnum])  
    b_fc2 = bias_variable([classnum])  
  
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  
  
    # loss   
  
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  
  
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
  
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))  
  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))  
  
    predict = [tf.reduce_max(y_conv),tf.argmax(y_conv,1)[0]]  
  
    saver = tf.train.Saver()  
    checkpoint_dir = modelpath+'train_model.cpkt'  
    if not os.path.exists(modelpath) or os.path.isfile(modelpath):  
    	os.makedirs(modelpath)  
    sess = tf.InteractiveSession()  
  
    sess.run(tf.initialize_all_variables())  
  
    file_object = open('thePredResults.txt','w')
    for idx in range(max_epoch):  
        imagess, labelss = next_batch(classnum*20)  
        if idx%(classnum*20) == 0:  
            acc = sess.run(accuracy,feed_dict={x:imagess,y_: labelss,keep_prob:1.0}) 
	    file_object.write('Epoch '+str(idx)+' is: '+str(acc)+'\n')  
            if acc == 1.0:  
            	break  
            print "At epoch %d, acc is %lf" % (idx,acc)  
        sess.run(train_step,feed_dict={x:imagess,y_: labelss,keep_prob:0.5}) 
 
    saver.save(sess,checkpoint_dir)  
    test_image,test_label = generate_test_image(900)  
    
    file_object.close()
    print "Total test acc: ", sess.run(accuracy,feed_dict={x:test_image,y_:test_label,keep_prob:1.0})  
  
if __name__ == '__main__':  
    read_image()  
    CNNmodel()  
