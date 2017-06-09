import lasagne
import theano
from lasagne import layers
from lasagne import regularization
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
import numpy as np
import theano.tensor as T
from scipy import misc
from scipy.stats import pearsonr
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.nonlinearities import rectify, sigmoid, identity, very_leaky_rectify, softmax
from sklearn.metrics import accuracy_score
import scipy.stats as st
import pickle
import sys
import os
import urllib
import gzip
import cPickle
import h5py
import math
import time
import pdb

from shape import ReshapeLayer
from my_canny import canny_edge
from unpool import Unpool2DLayer

# content in our ~/.theanorc:
#[global]
#floatX = float32
#
#[nvcc]
#fastmath = True

# RGB image
TrainImgeFolder = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/organized/train/img/';
# mask
TrainMaskFolder = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/organized/train/gt/';
# paired reg map (image-level prior)
TrainFCNFolder = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/organized/train/pairedr/';

# RGB image
TestImgeFolder = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/organized/test/img/';
# mask
TestMaskFolder = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/organized/test/gt/';
# paired reg map (image-level prior)
TestFCNFolder = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/organized/test/pairedr/';

# cache version code, intermediate data will be saved on disk as cache
TrainCode = 'cache_train_v1';
TestCode = 'cache_test_v1';

# CNN version code
CNNCode = 'V1';

# output folder
OutputFolder = 'prediction_output_v1';

LearningRate = 0.002;
WeightDecay = 0.00001;
BatchSize = 500;
Epochs = 600; #1000 more meaningfull?
GTType = np.float32;


def patch_extract(img, edge, mask):
    # selecting three points x1,y1 x2,y2 x3,y3
 
    if img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]:
        x_all, y_all = np.nonzero(mask[32/2:-32/2+1, 32/2:-32/2+1]);
        x_all = x_all + 32/2;
        y_all = y_all + 32/2;
        if len(x_all) <= 0:
            x1 = np.random.randint(32/2, img.shape[0]-32/2+1);
            y1 = np.random.randint(32/2, img.shape[1]-32/2+1);
        else:
            ind = np.random.randint(0, len(x_all));
            x1 = x_all[ind];
            y1 = y_all[ind];

    x2 = np.random.randint(32/2, img.shape[0]-32/2+1);
    y2 = np.random.randint(32/2, img.shape[1]-32/2+1);

    x_all, y_all = np.nonzero(edge[32/2:-32/2+1, 32/2:-32/2+1]);
    x_all = x_all + 32/2;
    y_all = y_all + 32/2;
    if len(x_all) <= 0:
        x3 = np.random.randint(32/2, img.shape[0]-32/2+1);
        y3 = np.random.randint(32/2, img.shape[1]-32/2+1);
    else:
        ind = np.random.randint(0, len(x_all));
        x3 = x_all[ind];
        y3 = y_all[ind];

    patch1 = img[x1-32/2:x1+32/2, y1-32/2:y1+32/2, :].transpose();
    truth1 = np.expand_dims(mask[x1-32/2:x1+32/2, y1-32/2:y1+32/2], axis=2).transpose();

    patch2 = img[x2-32/2:x2+32/2, y2-32/2:y2+32/2, :].transpose();
    truth2 = np.expand_dims(mask[x2-32/2:x2+32/2, y2-32/2:y2+32/2], axis=2).transpose();

    patch3 = img[x3-32/2:x3+32/2, y3-32/2:y3+32/2, :].transpose();
    truth3 = np.expand_dims(mask[x3-32/2:x3+32/2, y3-32/2:y3+32/2], axis=2).transpose();

    return patch1, truth1, patch2, truth2, patch3, truth3;


'''
def patch_extract(img, edge, mask):
    source = np.random.random();
    if source < 0.25 and img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]:
        x_all, y_all = np.nonzero(mask[32/2:-32/2+1, 32/2:-32/2+1]);
        x_all = x_all + 32/2;
        y_all = y_all + 32/2;
        if len(x_all) <= 0:
            x = np.random.randint(32/2, img.shape[0]-32/2+1);
            y = np.random.randint(32/2, img.shape[1]-32/2+1);
        else:
            ind = np.random.randint(0, len(x_all));
            x = x_all[ind];
            y = y_all[ind];
    elif source < 0.5:
        x = np.random.randint(32/2, img.shape[0]-32/2+1);
        y = np.random.randint(32/2, img.shape[1]-32/2+1);
    else:
        x_all, y_all = np.nonzero(edge[32/2:-32/2+1, 32/2:-32/2+1]);
        x_all = x_all + 32/2;
        y_all = y_all + 32/2;
        if len(x_all) <= 0:
            x = np.random.randint(32/2, img.shape[0]-32/2+1);
            y = np.random.randint(32/2, img.shape[1]-32/2+1);
        else:
            ind = np.random.randint(0, len(x_all));
            x = x_all[ind];
            y = y_all[ind];

    patch = img[x-32/2:x+32/2, y-32/2:y+32/2, :].transpose();
    truth = np.expand_dims(mask[x-32/2:x+32/2, y-32/2:y+32/2], axis=2).transpose();

    return patch, truth;
'''

def patches_extract_all(Data):
    X, M, E, _ = Data;
    nPatches = 3*len(X);
    
    ind = np.arange(len(X));
    np.random.shuffle(ind);

    si = 0;
    exc = ind;

    Xr = np.empty(shape=(nPatches, 4, 32, 32), dtype=np.float32);
    y = np.empty(shape=(nPatches, 1, 32, 32), dtype=GTType);

    i = 0;
    for bi in range(0,nPatches,3):
        p1,y1,p2,y2,p3,y3 = patch_extract(X[exc[i]], E[exc[i]], M[exc[i]]);
        Xr[bi, :, :, :], y[bi, :, :, :] = p1, y1;
        Xr[bi+1, :, :, :], y[bi+1, :, :, :] = p2, y2;
        Xr[bi+2, :, :, :], y[bi+2, :, :, :] = p3, y3;
        i = i+1;

    yield Xr, y;


'''
def patches_extract_all(Data):
    X, M, E, _ = Data;

    ind = np.arange(len(X));
    np.random.shuffle(ind);

    si = 0;
    exc = ind;

    Xr = np.empty(shape=(len(X), 4, 32, 32), dtype=np.float32);
    y = np.empty(shape=(len(X), 1, 32, 32), dtype=GTType);

    for bi in range(len(X)):
        Xr[bi, :, :, :], y[bi, :, :, :] = patch_extract(X[exc[bi]], E[exc[bi]], M[exc[bi]]);
    yield Xr, y;
'''

def data_aug(X, y):
    bs = X.shape[0];
    h_indices = np.random.choice(bs, bs / 2, replace = False);  # horizontal flip
    v_indices = np.random.choice(bs, bs / 2, replace = False);  # vertical flip
    r_indices = np.random.choice(bs, bs / 2, replace = False);  # 90 degree rotation

    X[h_indices] = X[h_indices, :, :, ::-1];
    X[v_indices] = X[v_indices, :, ::-1, :];
    y[h_indices] = y[h_indices, :, :, ::-1];
    y[v_indices] = y[v_indices, :, ::-1, :];
    for rot in range(np.random.randint(3) + 1):
        X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3);
        y[r_indices] = np.swapaxes(y[r_indices, :, :, :], 2, 3);

    return X, y;

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel;


def cnn_pred_mask(cnn, Test):
    
    if not os.path.exists('./' + OutputFolder + '/'):
        os.mkdir('./' + OutputFolder + '/');
        
    X, M, _, F = Test;
    for im_id in range(len(X)):
        if os.path.exists('./' + OutputFolder + '/' + F[im_id][:-4] + '.png'):
            continue;

        # print 'begin predict...';
        # print 'image: ', F[im_id][:];

        img = X[im_id];
        pred_m = np.zeros_like(M[im_id], dtype=np.float32);
        
        Map = img[:,:,3];
        colSize = Map.shape[1];
        rowSize = Map.shape[0];
	
        V = list(Map.flat);
        sorted_idx = np.argsort(V);
        
        i = 0; ## i--begining of segment
        while(i < len(V)):
            count = V.count(V[sorted_idx[i]]);
            st=i; en=st+count;
        	
            ## get 10% of the points in segment
            '''nPoints = int(math.ceil(0.1*count)); 
                selected_points = np.random.randint(st, high=en, size=nPoints);
            '''
            ## get the middle point of each segment
            ## ???? check if it has a bad result if the 32x32 patch of the point has overlap with other segments ????
            mid_p_t = time.time();

            mid_point_seg = int(math.floor(st+count/2.0));
            mid_point_seg_idx = sorted_idx[mid_point_seg];
            mx = mid_point_seg_idx/colSize; my = mid_point_seg_idx%colSize;
            mx = max(mx, 32/2); my = max(my, 32/2);
            mx = min(mx, rowSize-32/2); my = min(my, colSize-32/2);
            net_input = [img[mx-32/2:mx+32/2, my-32/2:my+32/2, :].transpose()];
            cnn_st = time.time();
            predict_patch = cnn.predict(net_input).reshape((-1, 32, 32)); # predict_patch[x-32/2:x+32/2, y-32/2:y+32/2]

            #print 'cnn took: ', time.time()-cnn_st, 'sec.';
            #print 'getting mid point+cnn took: ', time.time()-cnn_st, 'sec.';

            meanPoints = np.mean(predict_patch[0, :, :]);

            '''mean10p_t = time.time();
                ## get mean of 10% of the points
                sumPoints = 0;
                for j in range(len(selected_points)):
                    real_idx = sorted_idx[selected_points[j]];
                    x = real_idx/colSize; y = real_idx%colSize;
                    x = max(x, 32/2); y = max(y, 32/2);
                    x = min(x, rowSize-32/2); y = min(y, colSize-32/2);
                net_input = [img[x-32/2:x+32/2, y-32/2:y+32/2, :].transpose()];
                    predict_patch = cnn.predict(net_input).reshape((-1, 32, 32)); # predict_patch[x-32/2:x+32/2, y-32/2:y+32/2]
                    sumPoints = sumPoints + predict_patch[0, 32/2, 32/2];
            
                meanPoints1 = float(sumPoints)/nPoints;
                
            print 'getting mean of 10% of points took: ', time.time()-mean10p_t, 'sec.';
            '''
            ##put mean val within all segment
            for j in range(st,en):
                real_idx = sorted_idx[j];
                x = real_idx/colSize; y = real_idx%colSize;
                pred_m[x, y] = meanPoints;
       
            ## next segment
            i = i+count;
        #end while

        #pdb.set_trace()

	    go_line_t = time.time();
	
        ## going through lines of segments, with window 32x32, non overlapped, put the prediction of cnn for window.
        check = np.zeros((rowSize, colSize), dtype=bool); #for checking overlap
        threshold = 0.2*np.amax(pred_m);
        #print 'max prediction pixel:', np.amax(pred_m);
        point_seen_count = 0;
        for i in range(32/2, rowSize-32/2+1):
            for j in range(32/2, colSize-32/2+1):
                if(check[i,j] == 1):
                    continue;

                i_d = min(i+1, rowSize-32/2); j_d = j;
                i_r = i; j_r = min(j+1, colSize-32/2);
                if(pred_m[i,j] < threshold and pred_m[i_d,j_d] < threshold and pred_m[i_r,j_r] < threshold):
                    continue;

                if(pred_m[i,j] != pred_m[i_d,j_d] or pred_m[i,j] != pred_m[i_r,j_r]):
                    point_seen_count = point_seen_count+1;
                    net_input = [img[i-32/2:i+32/2, j-32/2:j+32/2, :].transpose()]; #######################FIXME# it can get n patches to predict #################################
                    predict_patch = cnn.predict(net_input).reshape((-1, 32, 32)); # 1x32x32

                    #pred_m[i-32/2:i+32/2, j-32/2:j+32/2] = np.mean([predict_patch[0,:,:], pred_m[i-32/2:i+32/2, j-32/2:j+32/2]], axis=0);
                    x_d = [0,1,-1, 1,-1,1,0,-1, 0];
                    y_d = [0,1,-1,-1, 1,0,1, 0,-1];
                    sum_neigh = 0.0;
                    for d in range(9):
                        sum_neigh += predict_patch[0, 32/2+x_d[d], 32/2+y_d[d]];
                    mean_neigh = sum_neigh/9.0;

                    for d in range(9):
                        pred_m[i+x_d[d],j+y_d[d]] = mean_neigh;
                        check[i+x_d[d],j+y_d[d]] = 1;

        print 'image: ', F[im_id][:];
        print 'number of runned CNN: ', point_seen_count;
        print 'going through lines took (cnn for each patch): ', time.time()-go_line_t, 'sec.';
        #save output result
        misc.imsave('./' + OutputFolder + '/' + F[im_id][:-4] + '.png', pred_m);
        

'''def cnn_pred_mask(cnn, Test):
    if not os.path.exists('./' + OutputFolder + '/'):
        os.mkdir('./' + OutputFolder + '/');

    gsm = gkern(32, 32/4);
    X, M, _, F = Test;
    for im_id in range(len(X)):
        if os.path.exists('./' + OutputFolder + '/' + F[im_id][:-4] + '.png'):
            continue;
        img = X[im_id];
        pred_m = np.zeros_like(M[im_id], dtype=np.float32);
        num_m = np.zeros_like(M[im_id], dtype=np.float32);

        for mir in range(2):
            img = img[::-1, :, :]
            pred_m = pred_m[::-1, :];
            num_m = num_m[::-1, :];
            for rot in range(4):
                img = np.swapaxes(img, 0, 1)[::-1, :, :];
                pred_m = np.swapaxes(pred_m, 0, 1)[::-1, :];
                num_m = np.swapaxes(num_m, 0, 1)[::-1, :];
                for x in range(32/2, pred_m.shape[0]-32/2+1):
                    net_inputs = np.zeros(shape=(pred_m.shape[1]-32+1, img.shape[2], 32, 32), dtype=np.float32);
                    yind = 0;
                    for y in range(32/2, pred_m.shape[1]-32/2+1):
                        net_inputs[yind, :, :, :] = img[x-32/2:x+32/2, y-32/2:y+32/2, :].transpose();
                        yind += 1;

                    net_outputs = cnn.predict(net_inputs).reshape((-1, 32, 32));
                    yind = 0;
                    for y in range(32/2, pred_m.shape[1]-32/2+1):
                        pred_m[x-32/2:x+32/2, y-32/2:y+32/2] += net_outputs[yind, :, :].transpose() * gsm;
                        num_m[x-32/2:x+32/2, y-32/2:y+32/2] += gsm;
                        yind += 1;

        pred_m /= num_m;
        misc.imsave('./' + OutputFolder + '/' + F[im_id][:-4] + '.png', pred_m);
'''

def load_data(image_folder, mask_folder, fcn_folder):
    X = [];
    M = [];
    E = [];
    F = [];

    for fn in os.listdir(image_folder):
        if os.path.isfile(image_folder + fn):
            if not os.path.exists(mask_folder + 'SHL_GT' + fn[3:-4] + '.png'):
                continue;
            if not os.path.exists(fcn_folder + fn[:-4] + '.png'):
                continue;
            im = misc.imread(image_folder + fn);
            im_meaned = np.zeros(shape=(im.shape[0], im.shape[1], 4), dtype=np.float32);
            im_meaned[:, :, 0] = (im[:, :, 0].astype(np.float32) - 126.61) / 100.0;
            im_meaned[:, :, 1] = (im[:, :, 1].astype(np.float32) - 123.21) / 100.0;
            im_meaned[:, :, 2] = (im[:, :, 2].astype(np.float32) - 112.63) / 100.0;

            fcn = misc.imread(fcn_folder + fn[:-4] + '.png').astype(np.float32);
            im_meaned[:, :, 3] = (fcn - 91.80) / 100.0;

            edge = canny_edge(im);
            mask = misc.imread(mask_folder + 'SHL_GT' + fn[3:-4] + '.png');

            if im_meaned.shape[0] == mask.shape[0] and im_meaned.shape[1] == mask.shape[1]:
                X.append(im_meaned);
                E.append((edge > 0).astype(np.uint8));
                M.append((mask > 0).astype(np.uint8));
                F.append(fn);

    print "Data Loaded", len(X);
    return [X, M, E, F];

#############################################################
# main starts
sys.setrecursionlimit(10000);

# load training and testing data
if os.path.exists('./data_cache/' + TrainCode + '.pkl'):
    if not os.path.exists('./data_cache/cnn_' + CNNCode + '-' + TrainCode + '.pkl'):
        Train = pickle.load(open('./data_cache/' + TrainCode + '.pkl', 'rb'));
else:
    Train = load_data(TrainImgeFolder, TrainMaskFolder, TrainFCNFolder);
    pickle.dump(Train, open('./data_cache/' + TrainCode + '.pkl', 'w'));

if os.path.exists('./data_cache/' + TestCode + '.pkl'):
    Test = pickle.load(open('./data_cache/' + TestCode + '.pkl', 'rb'));
else:
    Test = load_data(TestImgeFolder, TestMaskFolder, TestFCNFolder);
    pickle.dump(Test, open('./data_cache/' + TestCode + '.pkl', 'w'));

# build CNN architecture
cnn = NeuralNet(
    layers=[
        ('input', layers.InputLayer),

        ('conv1', layers.Conv2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('nin1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),

        ('conv3', layers.Conv2DLayer),
        ('conv4', layers.Conv2DLayer),
        ('nin2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),

        ('conv5', layers.Conv2DLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 4, 32, 32),

    conv1_num_filters=50,
    conv1_filter_size=(3, 3),
    conv1_nonlinearity=rectify,

    conv2_num_filters=50,
    conv2_filter_size=(3, 3),
    conv2_nonlinearity=rectify,

    nin1_num_filters=30,
    nin1_filter_size=(1, 1),
    nin1_nonlinearity=rectify,

    pool1_pool_size=(2, 2),
    pool1_stride=(2, 2),

    conv3_num_filters=50,
    conv3_filter_size=(3, 3),
    conv3_nonlinearity=rectify,

    conv4_num_filters=50,
    conv4_filter_size=(3, 3),
    conv4_nonlinearity=rectify,

    nin2_num_filters=30,
    nin2_filter_size=(1, 1),
    nin2_nonlinearity=rectify,

    pool2_pool_size=(2, 2),
    pool2_stride=(2, 2),

    conv5_num_filters=50,
    conv5_filter_size=(3, 3),
    conv5_nonlinearity=rectify,

    output_num_units=32*32,
    output_nonlinearity=sigmoid,

    update_learning_rate=LearningRate,
    update_momentum=0.975,
    objective_loss_function=lasagne.objectives.binary_crossentropy,
    batch_iterator_train=BatchIterator(batch_size=100),
    batch_iterator_test=BatchIterator(batch_size=100),
    train_split=TrainSplit(eval_size=0.05),
    regression=True,
    max_epochs=1,
    verbose=1,
);

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

# cnn init
cnn.load_params_from('./model_trained_on_UCF.pkl');

if os.path.exists('./data_cache/cnn_' + CNNCode + '-' + TrainCode + '.pkl'):
    # check if a trained model already exists
    cnn.load_params_from('./data_cache/cnn_' + CNNCode + '-' + TrainCode + '.pkl');
else:
    cnn_train = time.time();
    # training a new model
    for epoch in range(Epochs):
        # for every epoch
        for batch in patches_extract_all(Train):
            # for every batch
            inputs, targets = batch;
            # data augmentation
            inputs, targets = data_aug(inputs, targets);
            # run cnn.fit for 1 iteration
            cnn_fit = time.time();
            cnn.fit(inputs, targets.reshape((-1, 1 * 32 * 32)));
            # print 'fitting cnn took: ', time.time()-cnn_fit, 'sec.';
        # for every 10 epoch, print testing accuracy
        '''if epoch % 10 == 0:
            for batch in patches_extract_all(Test):
                inputs, targets = batch;
                predicts = cnn.predict(inputs);
                T = targets.reshape((-1, 1 * 32 * 32)).flatten().astype(np.int32);
                P = (predicts.flatten() > 0.5).astype(np.int32);
                print("======================= {:.4f} =======================".format(accuracy_score(T, P)));'''
    print 'training cnn took: ', time.time()-cnn_train, 'sec.';
    # save the trained model
    cnn.save_params_to('./data_cache/cnn_' + CNNCode + '-' + TrainCode + '.pkl');

# save predicted shadow mask on the test set
cnn_test = time.time();
cnn_pred_mask(cnn, Test);
print 'testing cnn took: ', time.time()-cnn_test, 'sec.';

