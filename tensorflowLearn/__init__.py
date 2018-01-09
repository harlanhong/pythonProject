import tensorflow as tf
import tensorlayer as tl

if __name__ == '__main__':
    sess = tf.InteractiveSession();

    x_train,y_train,x_val,y_val,x_test,y_test = None

    #设置输入
    x = tf.placeholder(tf.float32,shape=[50,184,184,3],name='x')
    y_ = tf.placeholder(tf.float32,shape=[50,184,184,1],name='y_')

    #设置网络
    network = tl.layers.InputLayer(inputs=x,name='input')
    network = tl.layers.DropoutLayer(network,keep=0.8)
    network = tl.layers.Conv3dLayer(network,act=tf.nn.relu,shape=[1,1,121,3,114],
                                    strides=[1,1,1,1,1],padding='SAME',name='cnn_layer1')#output(?,64,184,114)
    network =tl.layers.Conv3dLayer(network,act=tf.nn.relu,shape=[3,121,1,114,38],
                                   strides=[1,1,1,1,1],padding='SAEM',name='cnn_layers2')#output(?,64,64,38)
    network = tl.layers.Conv3dLayer(network,act=tf.nn.relu,shape=[38,1,1,38,1],
                                    strides=[1,1,1,1,1],padding='SAME',name='cnn_layer3')#output(?,64,64,1)
    network =tl.layers.Conv3dLayer(network,act=tf.nn.relu,shape=[1,16,16,1,512],
                                   strides=[1,1,1,1,1],padding='SAME',name='cnn_layer4')#output(?,49,49,512)
    network = tl.layers.DeConv3dLayer(network,act=tf.nn.relu,shape=[512,1,1,1,512],
                                      output_shape=[120,120,1],strides=[1,1,1,1,1],padding='SAME',name='deconv3d');

    y = network.outputs;

    #定义损失函数
    cost = tl.cost.cross_entropy(y_,y,name='loss')

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)

    tl.utils.fit(sess,network,train_op,cost,
                 x_train,y_train,x,y_,
                 batch_size=500,n_epoch=500,
                 print_freq=5,X_val=x_val,y_val=y_val,eval_train=False)

    tl.files.save_npz(network.all_params,name='model.npz')
    sess.close()


