import os, time, datetime
import tensorflow as tf
import nibabel as nib
from model import build_model

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

#Parametrizing hyper-parameters
flags.DEFINE_integer('batch_size', 4,
                     'Batch size for training')
flags.DEFINE_string('my_dataset_dir', None,
                    'Directory in which the dataset files are stored')
flags.DEFINE_string('my_log_dir', None,
                    'Directory in which tensorboard log files are stored')

batch_size = FLAGS.batch_size
my_dataset_dir = FLAGS.my_dataset_dir
my_log_dir = FLAGS.my_log_dir

#Loading and Slicing Data
train_set = tf.transpose(tf.expand_dims(tf.cast(nib.load(os.path.join(my_dataset_dir, 'tr_im.nii')).get_fdata()/1000.0, tf.float32), axis=0), [3,1,2,0]) #100 512 512 1
train_mask_set = tf.one_hot(tf.transpose(tf.cast(nib.load(os.path.join(my_dataset_dir, 'tr_mask.nii')).get_fdata(), tf.uint8), [2,0,1]), 4) #100 512 512 4
train_mask_set_unencoded = tf.transpose(tf.expand_dims(tf.cast(nib.load(os.path.join(my_dataset_dir, 'tr_mask.nii')).get_fdata(), tf.uint8), axis=0), [3,1,2,0]) #100 512 512 1

eval_set = train_set[-16:,:,:,:] #16 512 512 1 float32
eval_mask_set = train_mask_set[-16:,:,:,:] #16 512 512 4 uint8
eval_mask_set_unencoded = train_mask_set_unencoded[-16:,:,:,:] #16 512 512 1 uint8
train_set = train_set[:-16,:,:,:] #84 512 512 1 float32
train_mask_set = train_mask_set[:-16,:,:,:] #84 512 512 4 uint8
train_mask_set_unencoded = train_mask_set_unencoded[:-16,:,:,:] #84 512 512 1 uint8

#Median frequency balancing
freqs = tf.math.reduce_sum(train_mask_set, [0,1,2])
occurences = tf.math.reduce_sum(train_mask_set, [1,2])
occurences = tf.math.reduce_sum(tf.where(occurences>0, 1, occurences), 0)
freqs = tf.math.divide(freqs, occurences)
sorted = tf.sort(freqs).numpy()
median = tf.math.divide(tf.math.add(sorted[1], sorted[2]),2)
freqs = tf.math.divide(median, freqs) #Frequencies of all label occurences for calculating the median frequency balanced loss

def main(unused_argv):

    #Initialiazing model and optimizer
    final_model = build_model()
    final_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003)

    #Initializing Metrics and Display parameters
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('eval_accuracy')

    #Defining key epoch operations
    def weighted_cross_entropy_loss(y, pred, weights=freqs):
        y_cast = tf.cast(y, tf.float32)
        size = tf.cast(tf.size(y_cast), tf.float32)
        loss = tf.divide(tf.reduce_sum(tf.math.multiply(tf.reduce_sum(tf.math.multiply(tf.math.negative(tf.math.log(pred)), y_cast), [0,1,2]), weights)), size)
        return loss

    def train_step(model, optimizer, x_train, y_train, y_accuracy):
        with tf.GradientTape() as tape:
            pred = model(x_train)
            loss = weighted_cross_entropy_loss(y_train, pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y_accuracy, pred)

    def eval_step(model, x_eval, y_eval, y_accuracy, summary_writer, epoch):
        pred = model(x_eval)
        loss = weighted_cross_entropy_loss(y_eval, pred)

        eval_loss(loss)
        eval_accuracy(y_accuracy, pred)

        #Organizing Evaluation Images for Tensorboard Analysis
        predicted_images = tf.cast(tf.expand_dims(tf.math.argmax(pred, -1), axis=-1), tf.uint8)
        imgs_to_load = predicted_images[0:1,:,:,:]
        for i in range(16):
            if i==0:
                imgs_to_load = tf.concat([imgs_to_load, y_accuracy[0:1,:,:,:]], axis=0)
            else:
                img_set = tf.concat([predicted_images[i:i+1,:,:,:], y_accuracy[i:i+1,:,:,:]], axis=0)
                imgs_to_load = tf.concat([imgs_to_load, img_set], axis=0)
        #Creating differentiable grayscale pixel values for each class - 0,1,2,3
        imgs_to_load = tf.multiply(imgs_to_load, 80)

        #Log Evaluation Metrics and Display Images in Tensorboard
        with summary_writer.as_default():
            tf.summary.image('Eval Images', imgs_to_load, max_outputs=32, step=epoch)
            tf.summary.scalar('loss', eval_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', eval_accuracy.result(), step=epoch)

    #Initializing Summary Writers
    current = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = my_log_dir + current + '/train'
    eval_log_dir = my_log_dir + current + '/eval'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    for epoch in range(50):
        for i in range(int(84/batch_size)):
            #Train for every batch
            img_batch = train_set[int(batch_size*i):int(batch_size*(i+1)),:,:,:] #2 512 512 1
            label_batch = train_mask_set[int(batch_size*i):int(batch_size*(i+1)),:,:,:] #2 512 512 4
            label_accuracy_batch = train_mask_set_unencoded[int(batch_size*i):int(batch_size*(i+1)),:,:,:] #2 512 512 1

            train_step(final_model, final_optimizer, img_batch, label_batch, label_accuracy_batch)

        #Log training metrics
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        #Evaluate on evaluation set
        eval_step(final_model, eval_set, eval_mask_set, eval_mask_set_unencoded, eval_summary_writer, epoch)

        #Reset metrics at the end of an epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        eval_loss.reset_states()
        eval_accuracy.reset_states()

if __name__ == '__main__':
    tf.compat.v1.app.run()
