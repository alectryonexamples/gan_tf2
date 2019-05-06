import numpy as np
import tensorflow as tf

class GAN(object):
    """
    @brief      Class that contains GAN functions
    """

    def __init__(self, gen_model, disc_model, loss_type="orig", disc_lr=1e-3, gen_lr=1e-3):
        """
        @brief      Creates a GAN object

        @param gen_model    generator model of typetf.keras.Model 
        @param disc_model   discriminator model of typetf.keras.Model 
        @param loss_type    choose between 'orig' as described by original paper or 'wasserstein'
        """
        if gen_model.get_output_shape_at(0) != disc_model.get_input_shape_at(0):
            raise Exception("Generator output and discriminator shape not correct.")

        self.gen_model = gen_model
        self.disc_model = disc_model

        self.disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=disc_lr)
        self.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=gen_lr)

        self.loss_type = loss_type
        if self.loss_type == "orig":
            self.disc_loss = self.disc_loss_orig
            self.gen_loss = self.gen_loss_orig
        elif self.loss_type == "wasserstein":
            self.disc_loss = self.disc_loss_wasserstein
            self.gen_loss = self.gen_loss_wasserstein
        else:
            raise Exception("Not a valid loss type.")

    def disc_loss_orig(self, real_samples, noise_samples):
        """
        @brief      discriminator loss from original GAN paper
        
        @param      real_samples   (N by data_dim) tensor/numpy array
        @param      noise_samples  (N by noise_dim) tensor/numpy array
        """
        generator_samples = self.gen_model(noise_samples)

        logits_real = self.disc_model(real_samples)
        logits_gen = self.disc_model(generator_samples)
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(logits_real.shape), logits=logits_real))
        loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(logits_gen.shape), logits=logits_gen))

        loss = loss_real + loss_gen
        return loss

    def gen_loss_orig(self, noise_samples):
        """
        @brief      generator loss from original GAN paper
        
        @param      real_samples   (N by data_dim) tensor/numpy array
        @param      noise_samples  (N by noise_dim) tensor/numpy array
        """
        generator_samples = self.gen_model(noise_samples)
        logits_gen = self.disc_model(generator_samples)
        # loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(logits_gen.shape), logits=logits_gen))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(logits_gen.shape), logits=logits_gen))
        return loss

    def disc_loss_wasserstein(self, real_samples, noise_samples):
        """
        @brief      discriminator loss from improved wasserstein gan paper
        
        @param      real_samples   (N by data_dim) tensor/numpy array
        @param      noise_samples  (N by noise_dim) tensor/numpy array
        """
        generator_samples = self.gen_model(noise_samples)

        # helps prevent zero norm
        x_hat = generator_samples + tf.random.normal(generator_samples.shape, stddev=0.1)
        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            logits_real = self.disc_model(real_samples)
            logits_gen = self.disc_model(generator_samples)

            dist_loss = -tf.reduce_mean(logits_real) + tf.reduce_mean(logits_gen)

            logits_x_hat = self.disc_model(x_hat)
            sum_logits_x_hat = tf.reduce_sum(logits_x_hat)
        grads = tape.gradient(sum_logits_x_hat, x_hat)
        grad_norms = tf.norm(grads, axis=1)

        grad_weight = 10.
        grad_loss = -grad_weight * tf.reduce_mean(tf.square(grad_norms - 1))
        loss = dist_loss + grad_loss
        return loss

    def gen_loss_wasserstein(self, noise_samples):
        """
        @brief      generator loss from improved wasserstein gan paper
        
        @param      real_samples   (N by data_dim) tensor/numpy array
        @param      noise_samples  (N by noise_dim) tensor/numpy array
        """
        generator_samples = self.gen_model(noise_samples)
        logits_gen = self.disc_model(generator_samples)

        loss = -tf.reduce_mean(logits_gen)
        return loss

    @tf.function
    def train_disc(self, real_batch, noise_batch):
        """
        @brief      trains discriminator with one iteration of gradient descent 
        """ 
        with tf.GradientTape() as tape:
            disc_loss = self.disc_loss(real_batch, noise_batch)
        disc_grad = tape.gradient(disc_loss, self.disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.disc_model.trainable_variables))

        return disc_loss

    @tf.function
    def train_gen(self, noise_batch):
        """
        @brief      trains generator with one iteration of gradient descent 
        """ 
        with tf.GradientTape() as tape:
            gen_loss = self.gen_loss(noise_batch)
        gen_grad = tape.gradient(gen_loss, self.gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.gen_model.trainable_variables))
        return gen_loss



