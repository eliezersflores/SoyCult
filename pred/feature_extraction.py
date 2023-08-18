import numpy as np
import os
import scipy.io as sio
import sys
import tensorflow as tf

path_folds = os.path.join('..', 'data', 'seeds_folds')
folds = [fold for fold in sorted(os.listdir(path_folds))]
cultivars = [cultivar for cultivar in sorted(os.listdir(os.path.join(path_folds, folds[0])))]

path_feats = os.path.join('..', 'data', 'feats')
if not os.path.exists(path_feats):
    os.makedirs(path_feats)

cnns = []
cnns.append('densenet121')
cnns.append('densenet169')
cnns.append('densenet201')
cnns.append('efficientnetb0')
cnns.append('efficientnetb1')
cnns.append('efficientnetb2')
cnns.append('efficientnetb3')
cnns.append('efficientnetb4')
cnns.append('efficientnetb5')
cnns.append('efficientnetb6')
cnns.append('efficientnetb7')
cnns.append('inceptionresnetv2')
cnns.append('inceptionv3')
cnns.append('mobilenet')
cnns.append('mobilenetv2')
cnns.append('mobilenetv3large')
cnns.append('mobilenetv3small')
cnns.append('nasnetlarge')
cnns.append('nasnetmobile')
cnns.append('resnet101')
cnns.append('resnet101v2')
cnns.append('resnet152')
cnns.append('resnet152v2')
cnns.append('resnet50')
cnns.append('resnet50v2')
cnns.append('vgg16')
cnns.append('vgg19')
cnns.append('xception')

for cnn in cnns:

    path_cnn = os.path.join(path_feats, cnn)
    if not os.path.exists(path_cnn):
        os.makedirs(path_cnn)

    if cnn == 'densenet121':
        from tensorflow.keras.applications import DenseNet121 as convnet
        from tensorflow.keras.applications.densenet import preprocess_input
        img_side = 224
    elif cnn == 'densenet169':
        from tensorflow.keras.applications import DenseNet169 as convnet
        from tensorflow.keras.applications.densenet import preprocess_input
        img_side = 224
    elif cnn == 'densenet201':
        from tensorflow.keras.applications import DenseNet201 as convnet
        from tensorflow.keras.applications.densenet import preprocess_input
        img_side = 224
    elif cnn == 'efficientnetb0':
        from tensorflow.keras.applications import EfficientNetB0 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 224
    elif cnn == 'efficientnetb1':
        from tensorflow.keras.applications import EfficientNetB1 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 240
    elif cnn == 'efficientnetb2':
        from tensorflow.keras.applications import EfficientNetB2 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 260
    elif cnn == 'efficientnetb3':
        from tensorflow.keras.applications import EfficientNetB3 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 300
    elif cnn == 'efficientnetb4':
        from tensorflow.keras.applications import EfficientNetB4 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 380
    elif cnn == 'efficientnetb5':
        from tensorflow.keras.applications import EfficientNetB5 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 456
    elif cnn == 'efficientnetb6':
        from tensorflow.keras.applications import EfficientNetB6 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 528
    elif cnn == 'efficientnetb7':
        from tensorflow.keras.applications import EfficientNetB7 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 600
    elif cnn == 'inceptionresnetv2':
        from tensorflow.keras.applications import InceptionResNetV2 as convnet
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        img_side = 299
    elif cnn == 'inceptionv3':
        from tensorflow.keras.applications import InceptionV3 as convnet
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        img_side = 299
    elif cnn == 'mobilenet':
        from tensorflow.keras.applications import MobileNet as convnet
        from tensorflow.keras.applications.mobilenet import preprocess_input
        img_side = 224
    elif cnn == 'mobilenetv2':
        from tensorflow.keras.applications import MobileNetV2 as convnet
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'mobilenetv3large':
        from tensorflow.keras.applications import MobileNetV3Large as convnet
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        img_side = 224
    elif cnn == 'mobilenetv3small':
        from tensorflow.keras.applications import MobileNetV3Small as convnet
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        img_side = 224
    elif cnn == 'nasnetlarge':
        from tensorflow.keras.applications import NASNetLarge as convnet
        from tensorflow.keras.applications.nasnet import preprocess_input
        img_side = 331
    elif cnn == 'nasnetmobile':
        from tensorflow.keras.applications import NASNetMobile as convnet
        from tensorflow.keras.applications.nasnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet101':
        from tensorflow.keras.applications import ResNet101 as convnet
        from tensorflow.keras.applications.resnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet101v2':
        from tensorflow.keras.applications import ResNet101V2 as convnet
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'resnet152':
        from tensorflow.keras.applications import ResNet152 as convnet
        from tensorflow.keras.applications.resnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet152v2':
        from tensorflow.keras.applications import ResNet152V2 as convnet
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'resnet50':
        from tensorflow.keras.applications import ResNet50 as convnet
        from tensorflow.keras.applications.resnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet50v2':
        from tensorflow.keras.applications import ResNet50V2 as convnet
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'vgg16':
        from tensorflow.keras.applications import VGG16 as convnet
        from tensorflow.keras.applications.vgg16 import preprocess_input
        img_side = 224
    elif cnn == 'vgg19':
        from tensorflow.keras.applications import VGG19 as convnet
        from tensorflow.keras.applications.vgg19 import preprocess_input
        img_side = 224
    elif cnn == 'xception':
        from tensorflow.keras.applications import Xception as convnet
        from tensorflow.keras.applications.xception import preprocess_input
        img_side = 299

    model = convnet(include_top=False, weights='imagenet', pooling='avg')

    for fold in folds:

        print(f'With {cnn}, feature extraction from the images in {fold}...')

        path_src = os.path.join(path_folds, fold)
        path_dst = os.path.join(path_cnn, fold)
        if not os.path.exists(path_dst):
            os.makedirs(path_dst)

        dataset = tf.keras.preprocessing.image_dataset_from_directory(
                path_src, 
                labels='inferred', 
                label_mode='categorical',
                class_names=cultivars,
                color_mode='rgb',
                batch_size=32,
                image_size=(img_side,img_side),
                shuffle=False,
                validation_split=0,
        )

        batch_idx = 0
        for batch_images, batch_labels in dataset:
            batch_imges_pp = preprocess_input(batch_images)
            if batch_idx == 0:
                features = model.predict(batch_imges_pp)
                labels = batch_labels
            else:
                features = np.vstack([features, model.predict(batch_imges_pp)])
                labels = np.vstack([labels, batch_labels])
            batch_idx += 1

        sio.savemat(os.path.join(path_dst, 'data.mat'), {'features':features, 'labels':labels})

print('Done!')
