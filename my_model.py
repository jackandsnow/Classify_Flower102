import glob
import keras
import numpy as np
import os

from shutil import copyfile
from PIL import Image
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf

data_path = r'E:\Workspace\PyCharmProjects\FlowerData'
classes_path = r'E:\Workspace\PyCharmProjects\FlowerData\class'
train_record = r'E:\Workspace\PyCharmProjects\Flower102\tfrecords\train224.tfrecord'
test_record = r'E:\Workspace\PyCharmProjects\Flower102\tfrecords\test224.tfrecord'
category = {
    0: 'pink primrose',
    1: 'hard-leaved pocket orchid',
    2: 'canterbury bells',
    3: 'sweet pea',
    4: 'english marigold',
    5: 'tiger lily',
    6: 'moon orchid',
    7: 'bird of paradise',
    8: 'monkshood',
    9: 'globe thistle',
    10: 'snapdragon',
    11: "colt's foot",
    12: 'king protea',
    13: 'spear thistle',
    14: 'yellow iris',
    15: 'globe-flower',
    16: 'purple coneflower',
    17: 'peruvian lily',
    18: 'balloon flower',
    19: 'giant white arum lily',
    20: 'fire lily',
    21: 'pincushion flower',
    22: 'fritillary',
    23: 'red ginger',
    24: 'grape hyacinth',
    25: 'corn poppy',
    26: 'prince of wales feathers',
    27: 'stemless gentian',
    28: 'artichoke',
    29: 'sweet william',
    30: 'carnation',
    31: 'garden phlox',
    32: 'love in the mist',
    33: 'mexican aster',
    34: 'alpine sea holly',
    35: 'ruby-lipped cattleya',
    36: 'cape flower',
    37: 'great masterwort',
    38: 'siam tulip',
    39: 'lenten rose',
    40: 'barbeton daisy',
    41: 'daffodil',
    42: 'sword lily',
    43: 'poinsettia',
    44: 'bolero deep blue',
    45: 'wallflower',
    46: 'marigold',
    47: 'buttercup',
    48: 'oxeye daisy',
    49: 'common dandelion',
    50: 'petunia',
    51: 'wild pansy',
    52: 'primula',
    53: 'sunflower',
    54: 'pelargonium',
    55: 'bishop of llandaff',
    56: 'gaura',
    57: 'geranium',
    58: 'orange dahlia',
    59: 'pink-yellow dahlia?',
    60: 'cautleya spicata',
    61: 'japanese anemone',
    62: 'black-eyed susan',
    63: 'silverbush',
    64: 'californian poppy',
    65: 'osteospermum',
    66: 'spring crocus',
    67: 'bearded iris',
    68: 'windflower',
    69: 'tree poppy',
    70: 'gazania',
    71: 'azalea',
    72: 'water lily',
    73: 'rose',
    74: 'thorn apple',
    75: 'morning glory',
    76: 'passion flower',
    77: 'lotus',
    78: 'toad lily',
    79: 'anthurium',
    80: 'frangipani',
    81: 'clematis',
    82: 'hibiscus',
    83: 'columbine',
    84: 'desert-rose',
    85: 'tree mallow',
    86: 'magnolia',
    87: 'cyclamen ',
    88: 'watercress',
    89: 'canna lily',
    90: 'hippeastrum ',
    91: 'bee balm',
    92: 'ball moss',
    93: 'foxglove',
    94: 'bougainvillea',
    95: 'camellia',
    96: 'mallow',
    97: 'mexican petunia',
    98: 'bromelia',
    99: 'blanket flower',
    100: 'trumpet creeper',
    101: 'blackberry lily'
}

flower_types = 102
image_size = 224
batch_size = 100
total_size = 8189
test_size = 866
train_size = 7323
epochs = 100


# 保存模型在训练过程中的损失和识别率变化曲线
def save_history(history, path, str_acc):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    # 1920*1080 size
    fig = plt.figure(figsize=(19.2, 10.8))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    ax1.grid(True, linestyle='--', color='b')
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    ax2.grid(True, linestyle='--', color='b')
    plt.tight_layout()
    plt.savefig(path + '/history' + str(epochs) + '_' + str_acc + '.jpg')
    print('**************** Saving History Successfully ***************')


# 图片预处理
def preprocess():
    image_labels_path = os.path.join(data_path, 'imagelabels.mat')
    image_labels = loadmat(image_labels_path)['labels'][0] - 1
    files = sorted(glob.glob(os.path.join(data_path, 'jpg', '*.jpg')))
    labels = np.array([i for i in zip(files, image_labels)])

    cwd = os.getcwd()
    dir_name = os.path.join(data_path, 'class')
    cur_dir_path = os.path.join(cwd, dir_name)
    if not os.path.exists(cur_dir_path):
        os.mkdir(cur_dir_path)
    for i in range(0, 102):
        class_dir = os.path.join(cwd, dir_name, str(i))
        os.mkdir(class_dir)
    for label in labels:
        src = str(label[0])
        dst = os.path.join(cwd, dir_name, label[1], src.split(os.sep)[-1])
        copyfile(src, dst)

    print("******************* Preprocess Finished *********************")


# 制作TFRecord数据
def create_tfrecord():
    if os.path.exists(train_record) and os.path.exists(test_record):
        return
    train_writer = tf.python_io.TFRecordWriter(train_record)
    test_writer = tf.python_io.TFRecordWriter(test_record)
    for r, dirs, _ in os.walk(classes_path):
        # d is picture's label
        for d in dirs:
            for _, _, files in os.walk(os.path.join(r, d)):
                # files are pictures' names
                for i in range(0, len(files)):
                    img = Image.open(os.path.join(r, d, files[i]))
                    # 设置需要转换的图片大小
                    img = img.resize((image_size, image_size))
                    # 将图片转化为原生bytes
                    img_raw = img.tobytes()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'img_raw': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[img_raw])),
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[int(d)]))
                            }))
                    # 每一类花取10%张作为测试集
                    if i % 10 == 0:
                        test_writer.write(example.SerializeToString())
                    else:
                        train_writer.write(example.SerializeToString())
        train_writer.close()
        test_writer.close()
        break
    print('****************** Create Tfrecord File Finished ************************')


# 读取TFRecord数据
def read_tfrecord(record_file):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([record_file])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    img = features['img_raw']
    label = features['label']
    img = tf.decode_raw(img, tf.uint8)
    # 标准化到 0-1 之间
    img = tf.cast(img, tf.float32) * (1. / 255)
    image_data = tf.reshape(img, [image_size, image_size, 3])
    label = tf.cast(label, tf.int64)
    return image_data, label


# 加载模型进行测试
def test_model(model_name, x, y, batch):
    print('*************** Testing Model ********************')
    # one-hot
    y = keras.utils.to_categorical(y, num_classes=flower_types)
    model = keras.models.load_model(model_name)
    loss, acc = model.evaluate(x, y, batch_size=batch)
    print('loss is {:.4f}'.format(loss))
    print('acc is  {:.2f}%\n'.format(acc * 100))


# 获取花卉标签所以对应的名称
def get_flower_name(key):
    for k in category.keys():
        if key == k:
            return category.get(key)
    return 'UNKNOWN'


# 传入图片进行识别
def predict_picture(pic_name, model_name):
    print('****************** Predicting Picture *******************')
    img = Image.open(pic_name)
    img = img.resize((image_size, image_size))
    img_data = np.reshape(img.getdata(), (1, image_size, image_size, 3))
    img_data = img_data * (1.0 / 255)
    model = keras.models.load_model(model_name)
    predict = model.predict(img_data)
    result = np.transpose(predict).tolist()
    key = list(result).index(max(result))
    flower_name = get_flower_name(key)
    print('Picture name is ' + pic_name)
    print('Predict label is', key, ', and flower\'s name is', flower_name)


def alex_net(x_train, y_train, x_test, y_test):
    y_train = keras.utils.to_categorical(y_train, num_classes=flower_types)
    y_test = keras.utils.to_categorical(y_test, num_classes=flower_types)

    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(image_size, image_size, 3),
                     padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(flower_types, activation='softmax'))
    sgd = SGD(lr=1e-2, decay=1e-9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=0.1, batch_size=100, epochs=epochs)
    loss, acc = model.evaluate(x_test, y_test, batch_size=50)
    print('loss is {:.4f}'.format(loss) + ', acc is  {:.2f}%\n'.format(acc * 100))
    model_name = 'result/alex_model_epoch' + str(epochs) + '_' + str(round(acc * 100, 2)) + '.h5'
    model.save(model_name)
    save_history(history, 'result', str(round(acc * 100, 2)))
    # 清除session
    keras.backend.clear_session()
    return model_name


if __name__ == '__main__':
    # preprocess()
    create_tfrecord()
    train_data, train_label = read_tfrecord(train_record)
    test_data, test_label = read_tfrecord(test_record)

    # 使用shuffle_batch可以随机打乱输入
    train_x, train_y = tf.train.shuffle_batch(
        [train_data, train_label], batch_size=train_size,
        capacity=train_size, num_threads=4, min_after_dequeue=1000)

    test_x, test_y = tf.train.shuffle_batch(
        [test_data, test_label], batch_size=test_size,
        capacity=test_size, num_threads=2, min_after_dequeue=200)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            train_x, train_y = sess.run([train_x, train_y])
            test_x, test_y = sess.run([test_x, test_y])
            print(train_x.shape, train_y.shape)
            print(test_x.shape, test_y.shape)
            coord.request_stop()
            coord.join(threads)
            sess.close()

    train_y = train_y.reshape((train_size, 1))
    test_y = test_y.reshape((test_size, 1))
    model_dir = alex_net(train_x, train_y, test_x, test_y)
    # test model
    test_model(model_dir, test_x, test_y, 100)
    # predict picture
    pic = r'E:\Workspace\PyCharmProjects\FlowerData\class\39\image_04560.jpg'
    predict_picture(pic, model_dir)

    print('**************************** Program End ***************************')
