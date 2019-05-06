import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse
from dataset import *
from gan import GAN

def build_simple_mlp_model(input_dim, hidden_layers, output_dim, name):
    input_tensor = tf.keras.layers.Input(shape=(input_dim,))
    net = input_tensor
    for num_neurons in hidden_layers:
        net = tf.keras.layers.Dense(num_neurons)(net)
        net = tf.keras.layers.PReLU()(net)
    net = tf.keras.layers.Dense(output_dim, activation=None)(net)
    model = tf.keras.Model(inputs=input_tensor, outputs=net, name=name)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gan')
    parser.add_argument('-d', '--data_type', dest='data_type', type=str,
        choices=['quadratic', 'circle', 'sin', 'mnist'],
        default='quadratic',
        help='type of data to generate')
    parser.add_argument('--disc_iter', dest='disc_iter', type=int,
        default=2,
        help='number of discriminator training iterations per training cycle.')
    parser.add_argument('--gen_iter', dest='gen_iter', type=int,
        default=1,
        help='number of generator training iterations per training cycle')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
        default=32,
        help='training batch size.')
    parser.add_argument('--disc_model', dest='disc_model', nargs='+', type=int,
        default=[16, 16, 2],
        help='number of neurons in each hidden layer of discriminator model.')
    parser.add_argument('--gen_model', dest='gen_model', nargs='+', type=int,
        default=[16, 16, 2],
        help='number of neurons in each hidden layer of generator model.')
    parser.add_argument('--noise_dim', dest='noise_dim', type=int,
        default=5,
        help='dimension of noise for generator')
    parser.add_argument('--image_dir', dest='image_dir', type=str,
        default='images',
        help='directory to store images in.')
    parser.add_argument('--disc_lr', dest='disc_lr', type=float,
        default=1e-3,
        help='learning rate for discriminator.')
    parser.add_argument('--gen_lr', dest='gen_lr', type=float,
        default=1e-3,
        help='learning rate for discriminator.')
    args = parser.parse_args()

    if args.data_type == 'mnist':
        dataset = MNISTDataset(num=3, flatten=True)
    else:
        dataset = PointDataset(data_type=args.data_type)

    # build GAN
    gen_model = build_simple_mlp_model(args.noise_dim, args.gen_model, dataset.dim(), "gen")
    disc_model = build_simple_mlp_model(dataset.dim(), args.disc_model, 1, "disc")
    gan = GAN(gen_model, disc_model, loss_type="orig", disc_lr=args.disc_lr, gen_lr=args.gen_lr)

    # make image directory
    try:
        os.mkdir(args.image_dir)
    except FileExistsError as e:
        pass

    plt.figure(1, figsize=(7, 5))
    # train
    for i in range(10000):
        # train discriminator
        for _ in range(args.disc_iter):
            data_batch = dataset.sample(args.batch_size)
            noise_batch = tf.random.uniform((args.batch_size, args.noise_dim), dtype=np.float32)
            disc_loss = gan.train_disc(data_batch, noise_batch)

        # train generator
        for _ in range(args.gen_iter):
            noise_batch = tf.random.uniform((args.batch_size, args.noise_dim), dtype=np.float32)
            gen_loss = gan.train_gen(noise_batch)

        # show results every so often
        if i % 10 == 0:
            print("Iteration " + str(i) + ": ")
            print("Disc loss = " + str(disc_loss))
            print("Gen loss = " + str(gen_loss))

            if args.data_type == 'mnist':
                noise_batch = tf.random.uniform((15, args.noise_dim), dtype=np.float32)
                generator_samples = gen_model(noise_batch)

                plt.clf()
                plt.subplot(4, 4, 1)
                plt.imshow(np.reshape(data_batch[0, :], (28, 28)), vmin=0, vmax=1)
                plt.gca().set_axis_off()
                plt.title("Real")

                for j in range(15):
                    plt.subplot(4, 4, 2+j)
                    plt.imshow(np.reshape(generator_samples[j, :], (28, 28)), vmin=0, vmax=1)
                    plt.gca().set_axis_off()

                plt.show(block=False)
                plt.pause(0.1)
            else:
                noise_batch = tf.random.uniform((args.batch_size, args.noise_dim), dtype=np.float32)
                generator_samples = gen_model(noise_batch)
                plt.clf()
                plt.scatter(generator_samples[:, 0], generator_samples[:, 1], label='generated')
                plt.scatter(data_batch[:, 0], data_batch[:, 1], label="real")
                plt.legend(loc=3)
                plt.title("Iteration " + str(i))
                plt.show(block=False)
                plt.pause(0.1)
            plt.savefig(os.path.join(args.image_dir, str(i) + ".png"), bbox_inches='tight')

