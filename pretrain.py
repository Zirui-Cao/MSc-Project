import tensorflow as tf
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ##########  Use TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.TPUStrategy(resolver)

from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras_cv_attention_models.attention_layers import CompatibleExtractPatches
from data_loader import read_tfrecord_2d, read_tfrecord_3d


def load_datasets(batch_size, buffer_size,
                  tfrec_dir='gs://oai-challenge-us/tfrecords/',
                  ):
    """
    Loads tf records datasets for 2D models.
    """
    args = {
        'batch_size': batch_size,
        'buffer_size': buffer_size,
    }
    train_ds = read_tfrecord_2d(tfrecords_dir=os.path.join(tfrec_dir, 'train'),
                                batch_size=batch_size,
                                buffer_size=buffer_size,
                                augmentation=None,
                                is_training=True)
    valid_ds = read_tfrecord_2d(tfrecords_dir=os.path.join(tfrec_dir, 'valid'),
                                batch_size=batch_size,
                                buffer_size=buffer_size,
                                augmentation=None,
                                is_training=False)
    return train_ds, valid_ds


# tfrecords_dir = "gs://oai-challenge-us/tfrecords/train"
# file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
# print('file_list', file_list)
# tfrec_dir = 'gs://oai-challenge-us/tfrecords'
# train_ds, valid_ds = load_datasets(BATCH_SIZE, BUFFER_SIZE, tfrec_dir)
# print('train_ds',train_ds)
# print('valid_ds',valid_ds)


def parse_tf_img(element):
    image_feature_description = {
        "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "label_raw": tf.io.FixedLenFeature([], dtype=tf.string),
        "image_raw": tf.io.FixedLenFeature([], dtype=tf.string),
        "num_channels": tf.io.FixedLenFeature([], dtype=tf.int64),
        "height": tf.io.FixedLenFeature([], dtype=tf.int64), }
    parsed_example = tf.io.parse_single_example(element, image_feature_description)
    width = parsed_example['width']
    height = parsed_example['height']
    num_channels = parsed_example['num_channels']
    image = parsed_example['image_raw']
    image = tf.compat.v1.decode_raw(image, tf.float32)
    image = tf.reshape(image, [384, 384, 1])
    image = tf.cast(tf.round((image / tf.reduce_max(image)) * 255), tf.uint8)
    #   image = tf.cast((np.round(image)), tf.uint8)
    #   image = tf.cast((image-tf.reduce_mean(image)) / (tf.reduce_max(image)-tf.reduce_min(image)), tf.float32)
    #   image = tf.cast(image / tf.maximum(image)), tf.float32)
    mask = parsed_example['label_raw']
    mask = tf.compat.v1.decode_raw(mask, tf.int16)
    mask = tf.reshape(mask, [384, 384, 7])
    mask = tf.cast(mask, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 288, 288)
    mask = tf.image.resize_with_crop_or_pad(mask, 288, 288)

    return image


BUFFER_SIZE = 1024
BATCH_SIZE = 16
AUTO = tf.data.AUTOTUNE
# tfrecords_dir = "gs://oai-challenge-us/tfrecords/train"
# file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
# path_ds = tf.data.Dataset.from_tensor_slices(file_list)
# train_ds = path_ds.map(parse_tf_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
# test_ds = train_ds
# print('train_ds', train_ds)

train_ds, valid_ds = load_datasets(BATCH_SIZE, BUFFER_SIZE)
# iter = tf.compat.v1.data.make_one_shot_iterator(train_ds)
# next = iter.get_next()
# train_ds = next[0]
# a = train_ds.map(lambda img, mask: img)
print('train_ds', train_ds)
print('valid_ds', valid_ds)

# Setting seeds for reproducibility.
SEED = 42
tf.random.set_seed(SEED)
# keras.utils.set_random_seed(SEED)

# Set Parameters！！！！！！！！！！！！！！！
# DATA

INPUT_SHAPE = (288, 288, 1)
# INPUT_SHAPE = (512,512)
NUM_CLASSES = 10

# OPTIMIZER
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
# LEARNING_RATE = 5e-2
# WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 50

# AUGMENTATION
IMAGE_SIZE = 288  # We will resize input images to this size.
PATCH_SIZE = 16  # Size of the patches to be extracted from the input images.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0.50  # We have found 75% masking to give us the best results.

# ENCODER and DECODER
# LAYER_NORM_EPS = 1e-6
# ENC_PROJECTION_DIM = 256
# DEC_PROJECTION_DIM = 128
# ENC_NUM_HEADS = 6
# ENC_LAYERS = 4
# DEC_NUM_HEADS = 4
# DEC_LAYERS = (
#     2  # The decoder is lightweight but should be reasonably deep for reconstruction.
# )

LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 512
DEC_PROJECTION_DIM = 16
ENC_NUM_HEADS = 16
ENC_LAYERS = 12
DEC_NUM_HEADS = 12
DEC_LAYERS = (
    10  # The decoder is lightweight but should be reasonably deep for reconstruction.
)

ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]


# x_train = train_ds
# train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
# print('train_ds', train_ds)
# train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
# print('train_ds', train_ds)


# 提取补丁！！！！！！！！！！！！！！！
class Patches(layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

        # Assuming the image has three channels each patch would be
        # of size (patch_size, patch_size, 3).
        # self.resize = layers.Reshape((-1, patch_size * patch_size * 3))
        self.resize = layers.Reshape((-1, patch_size * patch_size))

    def build(self, input_shape):
        self.to_patch = CompatibleExtractPatches(
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID", )

    def call(self, images):
        # Create patches from the input images
        # print('Patches-images', images)
        # images=tf.reshape(images,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1])
        print('Patches-images', images)
        patches = self.to_patch(images)

        # Reshape the patches to (batch, num_patches, patch_area) and return it.
        patches = self.resize(patches)
        return patches


# 加入掩码！！！！！！！！！！！！！！！
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
            # tf.random.normal([1, patch_size * patch_size * 3]), trainable=True
            tf.random.normal([1, patch_size * patch_size]), trainable=True
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
        unmask_indices = rand_indices[:, self.num_mask:]
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


# 遮掩示例！！！！！！！！！！！！！！！
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
    inputs = layers.Input((None, ENC_PROJECTION_DIM))
    x = inputs
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
        x1 = x
        hidden_states_out.append(x1)

    print('Encoder x', x)
    outputs = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    return keras.Model(inputs, [outputs, hidden_states_out], name="mae_encoder")


# MAE Decoder！！！！！！！！！！！！！！！
def create_decoder(
        num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE
):
    inputs = layers.Input((NUM_PATCHES, ENC_PROJECTION_DIM))
    x = layers.Dense(DEC_PROJECTION_DIM)(inputs)
    print('Decoder x', x)

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    print('Decoder x', x)
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    print('Decoder x', x)

    ###########不flatten会有问题
    x = layers.Flatten()(x)
    print('x', x.shape)
    # x = layers.Dense(DEC_PROJECTION_DIM)(x)
    # pre_final = layers.Dense(units=image_size * image_size * 3, activation="sigmoid")(x)
    pre_final = layers.Dense(units=image_size * image_size * 1, activation="sigmoid")(x)
    # pre_final = layers.Dense(units=PATCH_SIZE * PATCH_SIZE * 1, activation="sigmoid")(x)
    # print('pre_final', pre_final)

    from einops import rearrange
    # outputs = rearrange(pre_final, 'b (h w) (c p1 p2) -> b (h p1) (w p2) c', p1=PATCH_SIZE, p2=PATCH_SIZE, h=IMAGE_SIZE//PATCH_SIZE, b=BATCH_SIZE)

    ##############网上找的恢复原来图像
    # print('pre_final', pre_final)
    # patches = tf.reshape(pre_final, [NUM_PATCHES, PATCH_SIZE, PATCH_SIZE, 1])
    # print('patches', patches)
    # reconstructed = tf.reshape(patches, [1, image_size, image_size, 1])
    # print('reconstructed', reconstructed)
    # rec_new = tf.nn.space_to_depth(reconstructed,2)
    # print('rec_new', rec_new)
    # outputs = tf.reshape(rec_new,[image_size,image_size,1])

    # outputs = layers.Reshape((NUM_PATCHES, PATCH_SIZE, PATCH_SIZE, 1))(pre_final)
    # print('outputs', outputs)
    # outputs = layers.Reshape((image_size, image_size, 1))(outputs)
    # print('outputs', outputs)
    # outputs = tf.nn.space_to_depth(outputs,2)
    # print('outputs', outputs)
    # outputs = layers.Reshape((image_size, image_size, 1))(outputs)

    # outputs = layers.Reshape((image_size, image_size, 3))(pre_final)
    outputs = layers.Reshape((image_size, image_size, 1))(pre_final)
    print('outputs', outputs)

    return keras.Model(inputs, outputs, name="mae_decoder")


# MAE Trainer！！！！！！！！！！！！！！！
class MaskedAutoencoder(keras.Model):
    def __init__(
            self,
            patch_layer,
            patch_encoder,
            encoder,
            decoder,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def calculate_loss(self, images, test=False):
        # Augment the input images.
        print('images', images)
        # Patch the augmented images.
        patches = self.patch_layer(images)
        print('patches', patches.shape)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs, _ = self.encoder(unmasked_embeddings)
        print('encoder_outputs', encoder_outputs.shape)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        # print('decoder_outputs', decoder_outputs[0])
        decoder_patches = self.patch_layer(decoder_outputs)

        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compiled_loss(loss_patch, loss_output)
        # print(total_loss)
        # total_loss1 = self.compiled_loss(augmented_images, decoder_outputs)

        return total_loss, loss_patch, loss_output

    def train_step(self, data):
        images, mask = data
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(images)

        # Apply gradients.
        train_vars = [
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        # print(loss_patch.dtype)
        # print(loss_output.dtype)
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, mask = data
        total_loss, loss_patch, loss_output = self.calculate_loss(images, test=True)

        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}


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


with strategy.scope():
    # 模型初始化！！！！！！！！！！！！！！！
    patch_layer = Patches()
    patch_encoder = PatchEncoder()
    encoder = create_encoder()
    decoder = create_decoder()

    mae_model = MaskedAutoencoder(
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        decoder=decoder,
    )

    # 训练callback！！！！！！！！！！！！！！！
    # 可视化callback
    # Taking a batch of test inputs to measure model's progress.
    test_images = next(iter(valid_ds))[0]


    # # test_images = test_ds.take(1)

    class TrainMonitor(keras.callbacks.Callback):
        def __init__(self, epoch_interval=None):
            self.epoch_interval = epoch_interval

        def on_epoch_end(self, epoch, logs=None):
            # if self.epoch_interval and epoch % self.epoch_interval == 0:
            if epoch == 5:

                # imgs_mask_test = mae_model.predict(test_images, verbose=1)
                # for i in range(imgs_mask_test.shape[0]):
                #     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
                #     ax[0].imshow(test_images[i])
                #     ax[0].set_title(f"Original: {epoch:03d}")
                #     ax[1].imshow(imgs_mask_test[i])
                #     ax[1].set_title(f"Resonstructed: {epoch:03d}")
                #     address = "/home/caozi/Result/" + str(i) + ".jpg"

                #     plt.savefig(address)
                #     plt.close(fig)

                test_patches = self.model.patch_layer(test_images)
                (
                    test_unmasked_embeddings,
                    test_masked_embeddings,
                    test_unmasked_positions,
                    test_mask_indices,
                    test_unmask_indices,
                ) = self.model.patch_encoder(test_patches)
                test_encoder_outputs, _ = self.model.encoder(test_unmasked_embeddings)
                test_encoder_outputs = test_encoder_outputs + test_unmasked_positions
                test_decoder_inputs = tf.concat(
                    [test_encoder_outputs, test_masked_embeddings], axis=1
                )
                test_decoder_outputs = self.model.decoder(test_decoder_inputs)

                # # Show a maksed patch image.
                # test_masked_patch, idx = self.model.patch_encoder.generate_masked_image(
                #     test_patches, test_unmask_indices
                # )
                # print(f"\nIdx chosen: {idx}")
                # original_image = test_augmented_images[idx]
                # masked_image = self.model.patch_layer.reconstruct_from_patch(
                #     test_masked_patch
                # )
                # reconstructed_image = test_decoder_outputs[idx]

                for i in range(test_patches.shape[0]):
                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
                    ax[0].imshow(test_images[i], cmap='gray')
                    ax[0].set_title(f"Original: {epoch:03d}")
                    ax[1].imshow(test_decoder_outputs[i], cmap='gray')
                    ax[1].set_title(f"Resonstructed: {epoch:03d}")
                    # address = "/rds/general/user/cc721/home/MAE1/Result/pitch_4/" + str(i + 1) + ".jpg"
                    # address = "D:/Imperial College London/BYSJ/CIFAR 10/Result/pitch_4/" + str(i + 1) + ".jpg"
                    address = "/home/caozi/Result/" + str(i) + ".jpg"

                    plt.savefig(address)
                    # address1 = "./Result/pitch_8/" + str(i + 1) +".jpg"
                    # cv2.imwrite(address1, test_augmented_images[i])
                    # address1 = "./Result/pitch_8/" + str(i + 1) + "_p.jpg"
                    # cv2.imwrite(address1, test_decoder_outputs[i])
                    # print(f"\nIdx chosen: {i}")
                    plt.close(fig)


    # nnn = len(x_train)
    nnn = 19200
    total_steps = int((nnn / BATCH_SIZE) * EPOCHS)
    warmup_epoch_percentage = 0.15
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    lrs = [scheduled_lrs(step) for step in range(total_steps)]

    # Assemble the callbacks.
    train_callbacks = [TrainMonitor(epoch_interval=96)]
    # train_callbacks = [TrainMonitor1(epoch_interval=5)]

    checkpoint_filepath = "/home/caozi/Model_save/checkpoint_0.50.h5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        verbose=1,
        monitor='loss',
        mode='min',
        save_best_only=True)

    # 模型编译与训练！！！！！！！！！！！！！！！
    optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)

    # Compile and pretrain the model.
    mae_model.compile(
        optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=["mae"]
    )

    # train_ds = strategy.experimental_distribute_dataset(train_ds)
    # valid_ds = strategy.experimental_distribute_dataset(valid_ds)
    # history = mae_model.fit(
    #     train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=train_callbacks,
    # )

# from glob import glob
# tfrec_dir = 'gs://oai-challenge-us/tfrecords/'
# print(len(glob(os.path.join(tfrec_dir, 'train/*'))))
# steps_per_epochh = len(glob(os.path.join(tfrec_dir, 'train/*'))) / (BATCH_SIZE)
steps_per_epoch = 19200 // BATCH_SIZE
validation_steps = 4480 // BATCH_SIZE
print(steps_per_epoch)
train_ds = strategy.experimental_distribute_dataset(train_ds)
valid_ds = strategy.experimental_distribute_dataset(valid_ds)
history = mae_model.fit(
    train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=valid_ds,
    validation_steps=validation_steps
    , callbacks=model_checkpoint_callback
    # , callbacks = train_callbacks
)
# mae_model.load_weights(checkpoint_filepath, by_name = True)


# def parse_tf_img(element):
#   image_feature_description = {
#     "width": tf.io.FixedLenFeature([], dtype=tf.int64),
#     "label_raw": tf.io.FixedLenFeature([], dtype=tf.string),
#     "image_raw": tf.io.FixedLenFeature([], dtype=tf.string),
#     "num_channels": tf.io.FixedLenFeature([], dtype=tf.int64),
#     "height": tf.io.FixedLenFeature([], dtype=tf.int64),}
#   parsed_example = tf.io.parse_single_example(element, image_feature_description)
#   width = parsed_example['width']
#   height = parsed_example['height']
#   num_channels = parsed_example['num_channels']
#   image = parsed_example['image_raw']
#   image = tf.compat.v1.decode_raw(image, tf.float32)
#   image = tf.reshape(image, [384, 384, 1])   
#   mask = parsed_example['label_raw']
#   mask = tf.compat.v1.decode_raw(mask, tf.int16)
#   mask = tf.reshape(mask, [384, 384, 7]) 
#   image = tf.image.resize_with_crop_or_pad(image, 288, 288)
#   mask = tf.image.resize_with_crop_or_pad(mask, 288, 288)

#   return image, mask

# def load_data(data, pre_train='True'):
#     if data == 'Train':
#         train_dsds = np.zeros([19200,288,288,1])
#         mask_dsds = np.zeros([19200,288,288,7])
#         train_ds = np.ones([160,288,288,1], dtype = float)
#         mask_ds = np.ones([160,288,288,7], dtype = int)
#         for i in range(120):
#             print(i)
#             if i < 10:
#                 train_tfr = 'gs://oai-challenge-us/tfrecords/train/00' + str(i) + '-of-119.tfrecords'
#             elif i <100:
#                 train_tfr = 'gs://oai-challenge-us/tfrecords/train/0' + str(i) + '-of-119.tfrecords'
#             else:
#                 train_tfr = 'gs://oai-challenge-us/tfrecords/train/' + str(i) + '-of-119.tfrecords'
#             raw_train_dataset = tf.data.TFRecordDataset(train_tfr)
#             train_dataset = raw_train_dataset.map(parse_tf_img)
#             j = 0
#             for image, mask in train_dataset:
#                 train_ds[j,:,:,:] = image.numpy()
#                 mask_ds[j,:,:,:] = mask.numpy()
#                 j = j + 1
#             train_dsds[i*160:((i+1)*160),:,:,:] = train_ds
#             mask_dsds[i*160:((i+1)*160),:,:,:] = mask_ds

#     elif data == 'Valid':
#         train_dsds = np.zeros([4480,288,288,1])
#         mask_dsds = np.zeros([4480,288,288,7])
#         train_ds = np.ones([160,288,288,1], dtype = float)
#         mask_ds = np.ones([160,288,288,7], dtype = int)
#         for i in range(28):
#             print(i)
#             if i < 10:
#                 train_tfr = 'gs://oai-challenge-us/tfrecords/valid/00' + str(i) + '-of-027.tfrecords'
#             else:
#                 train_tfr = 'gs://oai-challenge-us/tfrecords/valid/0' + str(i) + '-of-027.tfrecords'
#             raw_train_dataset = tf.data.TFRecordDataset(train_tfr)
#             train_dataset = raw_train_dataset.map(parse_tf_img)
#             j = 0
#             for image, mask in train_dataset:
#                 train_ds[j,:,:,:] = image.numpy()
#                 mask_ds[j,:,:,:] = mask.numpy()
#                 j = j + 1
#             train_dsds[i*160:((i+1)*160),:,:,:] = train_ds
#             mask_dsds[i*160:((i+1)*160),:,:,:] = mask_ds
#     if pre_train == True:
#         return train_dsds
#     else:
#         return train_dsds, mask_dsds


# AUTO = tf.data.AUTOTUNE
# test_img, test_mask = load_data(data = "Valid", pre_train = False)

# # imgs_mask_test = mae_model.predict(valid_ds, verbose=1, steps=validation_steps)
# imgs_test = mae_model.predict(test_img, verbose=1, steps=validation_steps)
# # for i in range(imgs_mask_test.shape[0]):
# for i in range(160):
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
#     ax[0].imshow(test_img[i], cmap = 'gray')
#     ax[0].set_title(f"Original")
#     ax[1].imshow(imgs_test[i], cmap = 'gray')
#     ax[1].set_title(f"Resonstructed")
#     address = "/home/caozi/Result/" + str(i) + ".jpg"

#     plt.savefig(address)
#     plt.close(fig)