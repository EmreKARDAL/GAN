import csv
import os
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from DataSet import DataSet
from model import GAN

image_path = './faces/'
model_file = './models/'
sample_path = './samples/'
tag_file = 'list_attr_celeba.csv'
testing_file = 'test.csv'

generate_num = 1
batch_size = 64
image_wsize = 64
image_hsize = 64
oimage_wsize = 178
oimage_hsize = 178
caption_size = 3
noise_size = 4096
max_epoch = 1000

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)


def train():
    tr = GAN(caption_size=caption_size, image_wsize=image_wsize, image_hsize=image_hsize, noise_size=noise_size)
    checkpoint = tf.train.Checkpoint(generator=tr.generator, discriminator=tr.discriminator, g_optimizer=tr.g_optim,
                                     d_optimizer=tr.d_optim, g_loss_meter=tr.g_loss_metrics,
                                     d_loss_meter=tr.d_loss_metrics)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, model_file, max_to_keep=1)
    checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    data = DataSet(imagepath=image_path, tagfile=tag_file, image_wsize=image_wsize, image_hsize=image_hsize)
    epoch = -1
    start = time.time()
    while epoch < max_epoch:
        real_images, caption = data.next_batch(batch_size=batch_size)
        real_images = tf.reshape(real_images, shape=(batch_size, image_wsize, image_hsize, 3))

        if epoch > 8:
            s = 0
            while s < 2:
                with tf.GradientTape() as gen_tape:
                    noise = np.random.uniform(-1, 1, [batch_size, noise_size])
                    fake_images = tr.generator([noise, caption], training=True)
                    fake_output = tr.discriminator([fake_images, caption], training=False)
                    gen_loss = tr.g_loss_fn(fake_output)

                generator_gradients = gen_tape.gradient(gen_loss, tr.generator.trainable_variables)
                tr.g_optim.apply_gradients(zip(generator_gradients, tr.generator.trainable_variables))
                tr.g_loss_metrics(gen_loss)
                s += 1
            with tf.GradientTape() as disc_tape:
                fake_images = tr.generator([noise, caption], training=False)
                fake_output = tr.discriminator([fake_images, caption], training=True)
                real_output = tr.discriminator([real_images, caption], training=True)
                disc_loss = tr.d_loss_fn(real_output, fake_output)

            discriminator_gradients = disc_tape.gradient(disc_loss, tr.discriminator.trainable_variables)
            tr.d_optim.apply_gradients(zip(discriminator_gradients, tr.discriminator.trainable_variables))
            tr.d_loss_metrics(disc_loss)
        else:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                noise = np.random.uniform(-1, 1, [batch_size, noise_size])
                fake_images = tr.generator([noise, caption], training=True)
                fake_output = tr.discriminator([fake_images, caption], training=True)
                real_output = tr.discriminator([real_images, caption], training=True)
                disc_loss = tr.d_loss_fn(real_output, fake_output)
                gen_loss = tr.g_loss_fn(fake_output)

            generator_gradients = gen_tape.gradient(gen_loss, tr.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, tr.discriminator.trainable_variables)

            tr.g_optim.apply_gradients(zip(generator_gradients, tr.generator.trainable_variables))
            tr.d_optim.apply_gradients(zip(discriminator_gradients, tr.discriminator.trainable_variables))
            tr.g_loss_metrics(gen_loss)
            tr.d_loss_metrics(disc_loss)

        if epoch != data.N_epoch:
            epoch = data.N_epoch
            template = '[{}/{}] D_loss={} G_loss={} time: {} sec'
            print(template.format(epoch, max_epoch, tr.d_loss_metrics.result(),
                                  tr.g_loss_metrics.result(), time.time() - start))
            tr.g_loss_metrics.reset_states()
            tr.d_loss_metrics.reset_states()
            ckpt_manager.save()
            start = time.time()
    return


def generate(rand=True):
    tr = GAN(caption_size=caption_size, image_wsize=image_wsize, image_hsize=image_hsize, noise_size=noise_size)
    checkpoint = tf.train.Checkpoint(generator=tr.generator, discriminator=tr.discriminator, g_optimizer=tr.g_optim,
                                     d_optimizer=tr.d_optim, g_loss_meter=tr.g_loss_metrics,
                                     d_loss_meter=tr.d_loss_metrics)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, model_file, max_to_keep=1)
    checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    captions = []
    noises = []
    with open(testing_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, line in enumerate(reader):
            noises.append(line[1:1 + noise_size])
            captions.append(np.array([1.0, 1.0, 1.0]).astype(np.float32))

    generated_images = []
    i = 0
    for vec, noise in zip(captions, noises):
        i += 1
        if i > generate_num:
            break
        if rand:
            noise = np.random.uniform(-1, 1, [1, noise_size])
            tmp = [0.0]
            caption = np.array([[-1.0, 1.0, 1.0]]).astype(np.float32)
            with open(testing_file, 'w') as f:
                writer = csv.writer(f)
                tmp = np.concatenate((tmp, noise[0]), axis=0)
                tmp = np.concatenate((tmp, caption[0]), axis=0)
                writer.writerow(tmp)
        else:
            noise = np.array([noise]).astype(np.float32)
            caption = np.array([vec]).astype(np.float32)
        generated_images.append(tr.generator([noise, caption], training=False))

    for i, images in enumerate(generated_images, start=1):
        for j, image in enumerate(images, start=1):
            img = Image.fromarray(((image.numpy() + 1) * 127.5).astype(np.uint8)).resize((oimage_wsize, oimage_hsize),
                                                                                         Image.LANCZOS)
            img.save(os.path.join(sample_path, 'sample_{}_{}.jpg'.format(i, j)))

    return


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == '-train':
        train()
    elif args[0] == '-generate':
        generate(rand=True)
