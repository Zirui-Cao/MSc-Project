from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf

import numpy as np
# import random
from Blocks.UNETR import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock

# Setting seeds for reproducibility.
SEED = 42
tf.random.set_seed(SEED)
# keras.utils.set_random_seed(SEED)

# 设定超参数！！！！！！！！！！！！！！！
BUFFER_SIZE = 1024
BATCH_SIZE = 2
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

# OPTIMIZER
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
# LEARNING_RATE = 5e-2
# WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 100

# AUGMENTATION
IMAGE_SIZE = 512  # We will resize input images to this size.
PATCH_SIZE = 16  # Size of the patches to be extracted from the input images.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0  # We have found 75% masking to give us the best results.

# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 12
DEC_PROJECTION_DIM = 12
ENC_NUM_HEADS = 4
ENC_LAYERS = 12
DEC_NUM_HEADS = 4
DEC_LAYERS = (
    3  # The decoder is lightweight but should be reasonably deep for reconstruction.
)
# LAYER_NORM_EPS = 1e-6
# ENC_PROJECTION_DIM = 256
# DEC_PROJECTION_DIM = 128
# ENC_NUM_HEADS = 8
# ENC_LAYERS = 6
# DEC_NUM_HEADS = 8
# DEC_LAYERS = (
#     8  # The decoder is lightweight but should be reasonably deep for reconstruction.
# )
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]


# load dataset！！！！！！！！！！！！！！！

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# x_train = x_train.astype('float32')
# y_train = x_train.astype('float32')
# x_test = x_train.astype('float32')
# y_test = x_train.astype('float32')
(x_train, y_train), (x_val, y_val) = (
    (x_train[:40000], y_train[:40000]),
    (x_train[40000:], y_train[40000:]),
)
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Testing samples: {len(x_test)}")


train_ds = tf.data.Dataset.from_tensor_slices(x_train)
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices(x_val)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices(x_test)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)

# imgs_train = np.load("D:/毕业设计程序/DATA/U-Net Train/b_trains.npy")
# imgs_mask_train = np.load("D:/毕业设计程序/DATA/U-Net Train/b_masks.npy")
# imgs_train = imgs_train.astype('float32')
# imgs_mask_train = imgs_mask_train.astype('float32')
# imgs_train /= 255
# imgs_mask_train /= 255
# imgs_mask_train[imgs_mask_train > 0.4] = 1
# imgs_mask_train[imgs_mask_train <= 0.4] = 0
# train_ds = imgs_train
# mask_ds = imgs_mask_train
# x_train = train_ds

# def data_preprocess_model():
#     model = keras.Sequential(
#         [
#             layers.experimental.preprocessing.Rescaling(1 / 255.0),
#             layers.experimental.preprocessing.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
#         ],
#         name="data_preprocessing",
#     )
#     return model


# extract patches！！！！！！！！！！！！！！！
class Patches(layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

        # Assuming the image has three channels each patch would be
        # of size (patch_size, patch_size, 3).
        self.resize = layers.Reshape((-1, patch_size * patch_size * 3))

    def call(self, images):
        # Create patches from the input images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Reshape the patches to (batch, num_patches, patch_area) and return it.
        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images, patches):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(keras.preprocessing.image.array_to_img(images[idx]))
        # plt.imshow(keras.utils.array_to_img(images[idx]))
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (self.patch_size, self.patch_size, 3))
            plt.imshow(keras.preprocessing.image.img_to_array(patch_img))
            # plt.imshow(keras.utils.img_to_array(patch_img))
            plt.axis("off")
        plt.show()

        # Return the index chosen to validate it outside the method.
        return idx

    # taken from https://stackoverflow.com/a/58082878/10319735
    def reconstruct_from_patch(self, patch):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(patch, (num_patches, self.patch_size, self.patch_size, 3))
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed


# visualize patch！！！！！！！！！！！！！！！
# # Get a batch of images.
# image_batch = next(iter(train_ds))
#
# # Augment the images.
# augmentation_model = get_train_augmentation_model()
# augmented_images = augmentation_model(image_batch)
# # augmented_images = image_batch
#
# # Define the patch layer.
# patch_layer = Patches()
#
# # Get the patches from the batched images.
# patches = patch_layer(images=augmented_images)
#
# # Now pass the images and the corresponding patches
# # to the `show_patched_image` method.
# random_index = patch_layer.show_patched_image(images=augmented_images, patches=patches)
#
# # Chose the same chose image and try reconstructing the patches
# # into the original image.
# image = patch_layer.reconstruct_from_patch(patches[random_index])
# plt.imshow(image)
# plt.axis("off")
# plt.show()

# add masks！！！！！！！！！！！！！！！
class PatchEncoder(layers.Layer):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        projection_dim=ENC_PROJECTION_DIM,
        mask_proportion=MASK_PROPORTION,
        downstream=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = tf.Variable(
            tf.random.normal([1, patch_size * patch_size * 3]), trainable=True
        )

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape

        # Create the projection layer for the patches.
        self.projection = layers.Dense(units=self.projection_dim)

        # Create the positional embedding layer.
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        # Get the positional embeddings.
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        # Embed the patches.

        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx

    # def generate_masked_image1(self, patches, unmask_indices, num):
    #     # Choose a random patch and it corresponding unmask index.
    #     # idx = np.random.choice(patches.shape[0])
    #     idx = num
    #     patch = patches[idx]
    #     unmask_index = unmask_indices[idx]
    #
    #     # Build a numpy array of same shape as patch.
    #     new_patch = np.zeros_like(patch)
    #
    #     # Iterate of the new_patch and plug the unmasked patches.
    #     count = 0
    #     for i in range(unmask_index.shape[0]):
    #         new_patch[unmask_index[i]] = patch[unmask_index[i]]
    #     return new_patch, idx

# visualize masks！！！！！！！！！！！！！！！
# # Create the patch encoder layer.
# patch_encoder = PatchEncoder()
#
# # Get the embeddings and positions.
# (
#     unmasked_embeddings,
#     masked_embeddings,
#     unmasked_positions,
#     mask_indices,
#     unmask_indices,
# ) = patch_encoder(patches=patches)
#
#
# # Show a maksed patch image.
# new_patch, random_index = patch_encoder.generate_masked_image(patches, unmask_indices)
#
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# img = patch_layer.reconstruct_from_patch(new_patch)
# # plt.imshow(keras.utils.array_to_img(img))
# plt.imshow(keras.preprocessing.image.array_to_img(img))
# plt.axis("off")
# plt.title("Masked")
# plt.subplot(1, 2, 2)
# img = augmented_images[random_index]
# # plt.imshow(keras.utils.array_to_img(img))
# plt.imshow(keras.preprocessing.image.array_to_img(img))
# plt.axis("off")
# plt.title("Original")
# plt.show()


# MLP！！！！！！！！！！！！！！！
def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# MAE Blocks！！！！！！！！！！！！！！！
def create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS):
    # image_size = IMAGE_SIZE
    # patch_size = PATCH_SIZE
    aaa = int(IMAGE_SIZE/PATCH_SIZE)
    bbb = int(PATCH_SIZE)*int(PATCH_SIZE)*3
    # inputs = layers.Input((None, ENC_PROJECTION_DIM))
    inputs = layers.Input(shape=(aaa*aaa, ENC_PROJECTION_DIM))
    # inputs = layers.Input(shape=(4, ENC_PROJECTION_DIM))
    x = inputs
    print('x',x.shape)
    hidden_states_out = []
    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=ENC_PROJECTION_DIM, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=ENC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
        x1 = layers.Flatten()(x1)
        x1 = layers.Dense(units=aaa * aaa * bbb, activation="sigmoid")(x1)
        x1 = layers.Reshape((aaa, aaa, bbb))(x1)
        hidden_states_out.append(x1)

    outputs = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(units=aaa * aaa * bbb, activation="sigmoid")(outputs)
    outputs = layers.Reshape((aaa, aaa, bbb))(outputs)

    # hidden_states_out = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(hidden_states_out)
    return keras.Model(inputs, [outputs,hidden_states_out], name="mae_encoder")

# def create_decoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS):
#     inputs = layers.Input((None, ENC_PROJECTION_DIM))






# MAE Trainer！！！！！！！！！！！！！！！
class MaskedAutoencoder(keras.Model):
    def __init__(
        self,
        patch_layer,
        patch_encoder,
        encoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder

        self.en1 = UnetrBasicBlock(feature_size=64, kernel_size=3)
        self.en2 = UnetrPrUpBlock(feature_size=128, kernel_size=3, num_layer=3, upsample_kernel_size=2)
        self.en3 = UnetrPrUpBlock(feature_size=256, kernel_size=3, num_layer=2, upsample_kernel_size=2)
        self.en4 = UnetrPrUpBlock(feature_size=512, kernel_size=3, num_layer=1, upsample_kernel_size=2)
        self.de5 = UnetrUpBlock(feature_size=512, kernel_size=3, upsample_kernel_size=2)
        self.de4 = UnetrUpBlock(feature_size=256, kernel_size=3, upsample_kernel_size=2)
        self.de3 = UnetrUpBlock(feature_size=128, kernel_size=3, upsample_kernel_size=2)
        self.de2 = UnetrUpBlock(feature_size=64, kernel_size=3, upsample_kernel_size=2)
        self.out = tf.keras.Sequential([
            layers.Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal'),
            layers.Conv2D(1, 1, activation = 'sigmoid'),
        ])


    def call(self, images):
        # Patch the augmented images.
        patches = self.patch_layer(images)
        print(images)
        print('images', images.shape)
        print('patches', patches.shape)

        # Encode the patches.
        # (
        #     unmasked_embeddings,
        #     masked_embeddings,
        #     unmasked_positions,
        #     mask_indices,
        #     unmask_indices,
        # ) = self.patch_encoder(patches)
        unmasked_embeddings = self.patch_encoder(patches)
        print('unmasked_embeddings', unmasked_embeddings.shape)
        # Pass the unmaksed patche to the encoder.
        [encoder_outputs,hidden_states_out] = self.encoder(unmasked_embeddings)
        print('encoder_outputs', encoder_outputs.shape)


        enc1 = self.en1(images)
        x2 = hidden_states_out[3]
        print('x2', x2.shape)
        print('hidden_states_out6', hidden_states_out[6].shape)
        enc2 = self.en2(x2)
        print('enc2', enc2.shape)
        x3 = hidden_states_out[6]
        enc3 = self.en3(x3)
        print('enc3', enc3.shape)
        x4 = hidden_states_out[9]
        enc4 = self.en4(x4)
        print('enc4', enc4.shape)
        dec4 = encoder_outputs
        dec3 = self.de5(dec4, enc4)
        dec2 = self.de4(dec3, enc3)
        dec1 = self.de3(dec2, enc2)
        out = self.de2(dec1, enc1)
        logits = self.out(out)
        print('logits', logits.shape)
        return logits

    # def train_step(self, images):
    #     with tf.GradientTape() as tape:
    #         total_loss, loss_patch, loss_output = self.call(images)
    #
    #     # Apply gradients.
    #     train_vars = [
    #         self.patch_layer.trainable_variables,
    #         self.patch_encoder.trainable_variables,
    #         self.encoder.trainable_variables,
    #         self.decoder.trainable_variables,
    #     ]
    #     grads = tape.gradient(total_loss, train_vars)
    #     tv_list = []
    #     for (grad, var) in zip(grads, train_vars):
    #         for g, v in zip(grad, var):
    #             tv_list.append((g, v))
    #     self.optimizer.apply_gradients(tv_list)
    #
    #     # Report progress.
    #     # print(loss_patch.dtype)
    #     # print(loss_output.dtype)
    #     self.compiled_metrics.update_state(loss_patch, loss_output)
    #     return {m.name: m.result() for m in self.metrics}
    #
    # def test_step(self, images):
    #     total_loss, loss_patch, loss_output = self.calculate_loss(images)
    #
    #     # Update the trackers.
    #     self.compiled_metrics.update_state(loss_patch, loss_output)
    #     return {m.name: m.result() for m in self.metrics}


# model init！！！！！！！！！！！！！！！
patch_layer = Patches()
patch_encoder = PatchEncoder(downstream=True)
encoder = create_encoder()

mae_model = MaskedAutoencoder(
    patch_layer=patch_layer,
    patch_encoder=patch_encoder,
    encoder=encoder,
)


# train callback！！！！！！！！！！！！！！！
# Taking a batch of test inputs to measure model's progress.
# test_images = next(iter(test_ds))



# class TrainMonitor(keras.callbacks.Callback):
#     def __init__(self, epoch_interval=None):
#         self.epoch_interval = epoch_interval
#
#     def on_epoch_end(self, epoch, logs=None):
#         # if self.epoch_interval and epoch % self.epoch_interval == 0:
#         if epoch == 96:
#             test_augmented_images = self.model.test_augmentation_model(test_images)
#             test_patches = self.model.patch_layer(test_augmented_images)
#             (
#                 test_unmasked_embeddings,
#                 test_masked_embeddings,
#                 test_unmasked_positions,
#                 test_mask_indices,
#                 test_unmask_indices,
#             ) = self.model.patch_encoder(test_patches)
#             test_encoder_outputs = self.model.encoder(test_unmasked_embeddings)
#             test_encoder_outputs = test_encoder_outputs + test_unmasked_positions
#             test_decoder_inputs = tf.concat(
#                 [test_encoder_outputs, test_masked_embeddings], axis=1
#             )
#             test_decoder_outputs = self.model.decoder(test_decoder_inputs)
#
#             # # Show a maksed patch image.
#             # test_masked_patch, idx = self.model.patch_encoder.generate_masked_image(
#             #     test_patches, test_unmask_indices
#             # )
#             # print(f"\nIdx chosen: {idx}")
#             # original_image = test_augmented_images[idx]
#             # masked_image = self.model.patch_layer.reconstruct_from_patch(
#             #     test_masked_patch
#             # )
#             # reconstructed_image = test_decoder_outputs[idx]
#
#             for i in range(test_patches.shape[0]):
#                 fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
#                 ax[0].imshow(test_augmented_images[i])
#                 ax[0].set_title(f"Original: {epoch:03d}")
#                 ax[1].imshow(test_decoder_outputs[i])
#                 ax[1].set_title(f"Resonstructed: {epoch:03d}")
#                 # address = "/rds/general/user/cc721/home/MAE1/Result/pitch_4/" + str(i + 1) + ".jpg"
#                 address = "D:/Imperial College London/BYSJ/CIFAR 10/Result/pitch_4/" + str(i + 1) + ".jpg"
#
#                 # address = "Z:/home/MAE1/Result/pitch_4/" + str(i + 1) + ".jpg"
#                 plt.savefig(address)
#                 # address1 = "./Result/pitch_8/" + str(i + 1) +".jpg"
#                 # cv2.imwrite(address1, test_augmented_images[i])
#                 # address1 = "./Result/pitch_8/" + str(i + 1) + "_p.jpg"
#                 # cv2.imwrite(address1, test_decoder_outputs[i])
#                 # print(f"\nIdx chosen: {i}")
#                 plt.close(fig)
#
#
#             # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
#             # ax[0].imshow(original_image)
#             # ax[0].set_title(f"Original: {epoch:03d}")
#             #
#             # ax[1].imshow(masked_image)
#             # ax[1].set_title(f"Masked: {epoch:03d}")
#             #
#             # ax[2].imshow(reconstructed_image)
#             # ax[2].set_title(f"Resonstructed: {epoch:03d}")
#             #
#             # ax[3].imshow(test_decoder_outputs[idx+1])
#             # ax[3].set_title(f"Resonstructed: {epoch:03d}")
#             #
#             # plt.show()
#             # plt.close()

# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


total_steps = int((len(x_train) / BATCH_SIZE) * EPOCHS)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

lrs = [scheduled_lrs(step) for step in range(total_steps)]
# plt.plot(lrs)
# plt.xlabel("Step", fontsize=14)
# plt.ylabel("LR", fontsize=14)
# plt.show()



from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
def dice_coef(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f * y_true_f) + keras.sum(y_pred_f * y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# Assemble the callbacks.
# train_callbacks = [TrainMonitor(epoch_interval=96)]
# train_callbacks = [TrainMonitor1(epoch_interval=5)]


optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)

# Compile and pretrain the model.
mae_model.compile(
    optimizer=optimizer, loss=dice_coef_loss, metrics=["mae"]
)
# mae_model.compile(
#     optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=["mae"]
# )
# history = mae_model.fit(
#     train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=train_callbacks,
# )
history = mae_model.fit(
    train_ds, epochs=EPOCHS, validation_data=val_ds,
)
# history = mae_model.fit(
#     train_ds, mask_ds,  epochs=EPOCHS,
# )

# Measure its performance.
# loss, mae = mae_model.evaluate(test_ds)
# print(f"Loss: {loss:.2f}")
# print(f"MAE: {mae:.2f}")






