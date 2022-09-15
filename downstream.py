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
from Blocks.UNETR import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock

LESS_RATIO = 1
EPOCHS = 100
BUFFER_SIZE = 1024
BATCH_SIZE = 64
AUTO = tf.data.AUTOTUNE


# linear = True
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
                                is_training=True,
                                multi_class=True,
                                less_ratio=LESS_RATIO)
    valid_ds = read_tfrecord_2d(tfrecords_dir=os.path.join(tfrec_dir, 'valid'),
                                batch_size=batch_size,
                                buffer_size=buffer_size,
                                augmentation=None,
                                is_training=False,
                                multi_class=True)
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
    #   image = tf.cast(tf.round((image / tf.reduce_max(image)) * 255), tf.uint8)
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
NUM_CLASSES = 7

# OPTIMIZER
# LEARNING_RATE = 5e-3
# WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# AUGMENTATION
IMAGE_SIZE = 288  # We will resize input images to this size.
PATCH_SIZE = 16  # Size of the patches to be extracted from the input images.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0.75  # We have found 75% masking to give us the best results.

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
        print('Patches-images', images.shape)
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

    print('Encoder x', x.shape)
    outputs = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    return keras.Model(inputs, [outputs, hidden_states_out], name="mae_encoder")


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


class MaskedAutoencoder_down(keras.Model):
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

        aaa = int(IMAGE_SIZE / PATCH_SIZE)
        self.encoder_after = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=LAYER_NORM_EPS),
            layers.Reshape((aaa, aaa, ENC_PROJECTION_DIM)),
        ])

        self.en1 = UnetrBasicBlock(feature_size=64, kernel_size=3)
        self.en2 = UnetrPrUpBlock(feature_size=128, kernel_size=3, num_layer=2, upsample_kernel_size=2)
        self.en3 = UnetrPrUpBlock(feature_size=256, kernel_size=3, num_layer=1, upsample_kernel_size=2)
        self.en4 = UnetrPrUpBlock(feature_size=512, kernel_size=3, num_layer=0, upsample_kernel_size=2)
        self.de5 = UnetrUpBlock(feature_size=512, kernel_size=3, upsample_kernel_size=2)
        self.de4 = UnetrUpBlock(feature_size=256, kernel_size=3, upsample_kernel_size=2)
        self.de3 = UnetrUpBlock(feature_size=128, kernel_size=3, upsample_kernel_size=2)
        self.de2 = UnetrUpBlock(feature_size=64, kernel_size=3, upsample_kernel_size=2)
        # self.out = tf.keras.Sequential([
        #     layers.Conv2D(NUM_CLASSES, 1, padding = "same", kernel_initializer='he_normal'),
        #     layers.Activation('sigmoid'),
        # ])
        self.out = tf.keras.Sequential([
            layers.Conv2D(NUM_CLASSES, 1, padding="same", kernel_initializer='he_normal'),
            layers.Activation('softmax'),
        ])

    def call(self, images):
        patches = self.patch_layer(images)
        print('images', images.shape)
        print('patches', patches.shape)
        unmasked_embeddings = self.patch_encoder(patches)
        print('unmasked_embeddings', unmasked_embeddings.shape)
        # Pass the unmaksed patche to the encoder.
        [encoder_outputs, hidden_states_out] = self.encoder(unmasked_embeddings)
        print('encoder_outputs', encoder_outputs.shape)
        # print('hidden_states_out', hidden_states_out)

        enc1 = self.en1(images)
        x2 = self.encoder_after(hidden_states_out[3])
        print('x2', x2.shape)
        print('hidden_states_out6', self.encoder_after(hidden_states_out[6]).shape)
        enc2 = self.en2(x2)
        print('enc2', enc2.shape)
        x3 = self.encoder_after(hidden_states_out[6])
        enc3 = self.en3(x3)
        print('enc3', enc3.shape)
        x4 = self.encoder_after(hidden_states_out[9])
        enc4 = self.en4(x4)
        print('enc4', enc4.shape)
        dec4 = self.encoder_after(encoder_outputs)
        dec3 = self.de5(dec4, enc4)
        dec2 = self.de4(dec3, enc3)
        dec1 = self.de3(dec2, enc2)
        out = self.de2(dec1, enc1)
        logits = self.out(out)
        print('logits', logits.shape)
        return logits

    def train_step(self, data):
        images, mask = data
        with tf.GradientTape() as tape:
            mask_pre = self.call(images)
            # loss = dice_coef_loss(mask, mask_pre)
            loss = tversky_crossentropy(mask, mask_pre)

        # train_vars = [
        #     self.patch_layer.trainable_variables,
        #     self.patch_encoder.trainable_variables,
        #     self.encoder.trainable_variables,
        #     self.encoder_after.trainable_variables,
        #     self.en1.trainable_variables,
        #     self.en2.trainable_variables,
        #     self.en3.trainable_variables,
        #     self.en4.trainable_variables,
        #     self.de5.trainable_variables,
        #     self.de4.trainable_variables,
        #     self.de3.trainable_variables,
        #     self.de2.trainable_variables,
        #     self.out.trainable_variables,
        # ]
        # grads = tape.gradient(loss, train_vars)
        # tv_list = []
        # for (grad, var) in zip(grads, train_vars):
        #     for g, v in zip(grad, var):
        #         tv_list.append((g, v))
        # self.optimizer.apply_gradients(tv_list)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(mask_pre, mask)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, mask = data
        mask_pre = self.call(images)
        loss = dice_coef_loss(mask, mask_pre)

        # Update the trackers.
        self.compiled_metrics.update_state(mask_pre, mask)
        return {m.name: m.result() for m in self.metrics}


from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef_eval(y_true, y_pred):
    y_true = tf.slice(y_true, [0, 0, 0, 1], [-1, -1, -1, 6])
    y_pred = tf.slice(y_pred, [0, 0, 0, 1], [-1, -1, -1, 6])

    dice = dice_coef(y_true, y_pred)

    return dice


def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : tensor containing target mask.
    y_pred : tensor containing predicted mask.
    alpha : real value, weight of '0' class.
    beta : real value, weight of '1' class.
    smooth : small real value used for avoiding division by zero error.
    Returns
    -------
    tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)

    return 1 - answer


def tversky_crossentropy(y_true, y_pred):
    tversky = tversky_loss(y_true, y_pred)
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    crossentropy = K.mean(crossentropy)

    return tversky + crossentropy


with strategy.scope():
    # 模型初始化！！！！！！！！！！！！！！！
    patch_layer = Patches()
    patch_encoder = PatchEncoder()
    patch_encoder.downstream = True
    encoder = create_encoder()

    mae_seg = MaskedAutoencoder_down(
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
    )

    # print('call', mae_seg(tf.random.normal(shape=[int(BATCH_SIZE/8), 288, 288, 1])))
    a = mae_seg(tf.random.normal(shape=[int(BATCH_SIZE / 8), 288, 288, 1]))

    checkpoint_filepath = "/home/caozi/Model_save/checkpoint_0.75.h5"
    # mae_seg.load_weights(checkpoint_filepath, by_name = True, skip_mismatch=True)
    # a = mae_seg.layers[3].get_weights()[0]
    # print('weights', a)
    mae_seg.load_weights(checkpoint_filepath, by_name=True)
    # b = mae_seg.layers[3].get_weights()[0]
    # print('weights', b)
    # print('weight_cha', a-b)
    # if linear == True:
    # mae_seg.layers[0].trainable = False
    # mae_seg.layers[1].trainable = False
    # mae_seg.layers[2].trainable = False

    # 训练callback！！！！！！！！！！！！！！！
    # 可视化callback
    # Taking a batch of test inputs to measure model's progress.
    # test_images = next(iter(valid_ds))[0]
    # # test_images = test_ds.take(1)
    # print('test_images', test_images.shape)

    # nnn = len(x_train)
    nnn = 19200 * LESS_RATIO
    total_steps = int((nnn / (BATCH_SIZE // 8)) * EPOCHS)
    warmup_epoch_percentage = 0.15
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    lrs = [scheduled_lrs(step) for step in range(total_steps)]

    # 模型编译与训练！！！！！！！！！！！！！！！
    optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)

    # Compile and pretrain the model.
    mae_seg.compile(
        optimizer=optimizer, loss=tversky_crossentropy, metrics=[tversky_crossentropy]
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

# print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))

steps_per_epoch = (19200 * LESS_RATIO) // BATCH_SIZE
validation_steps = 4480 // BATCH_SIZE
print(steps_per_epoch)
train_ds = strategy.experimental_distribute_dataset(train_ds)
valid_ds = strategy.experimental_distribute_dataset(valid_ds)
print(mae_seg.summary())
history = mae_seg.fit(
    train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=valid_ds,
    validation_steps=validation_steps
)


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
    mask = parsed_example['label_raw']
    mask = tf.compat.v1.decode_raw(mask, tf.int16)
    mask = tf.reshape(mask, [384, 384, 7])
    image = tf.image.resize_with_crop_or_pad(image, 288, 288)
    mask = tf.image.resize_with_crop_or_pad(mask, 288, 288)

    return image, mask


def load_data(data, pre_train='True'):
    if data == 'Train':
        train_dsds = np.zeros([19200, 288, 288, 1])
        mask_dsds = np.zeros([19200, 288, 288, 7])
        train_ds = np.ones([160, 288, 288, 1], dtype=float)
        mask_ds = np.ones([160, 288, 288, 7], dtype=int)
        for i in range(120):
            print(i)
            if i < 10:
                train_tfr = 'gs://oai-challenge-us/tfrecords/train/00' + str(i) + '-of-119.tfrecords'
            elif i < 100:
                train_tfr = 'gs://oai-challenge-us/tfrecords/train/0' + str(i) + '-of-119.tfrecords'
            else:
                train_tfr = 'gs://oai-challenge-us/tfrecords/train/' + str(i) + '-of-119.tfrecords'
            raw_train_dataset = tf.data.TFRecordDataset(train_tfr)
            train_dataset = raw_train_dataset.map(parse_tf_img)
            j = 0
            for image, mask in train_dataset:
                train_ds[j, :, :, :] = image.numpy()
                mask_ds[j, :, :, :] = mask.numpy()
                j = j + 1
            train_dsds[i * 160:((i + 1) * 160), :, :, :] = train_ds
            mask_dsds[i * 160:((i + 1) * 160), :, :, :] = mask_ds

    elif data == 'Valid':
        train_dsds = np.zeros([4480, 288, 288, 1])
        mask_dsds = np.zeros([4480, 288, 288, 7])
        train_ds = np.ones([160, 288, 288, 1], dtype=float)
        mask_ds = np.ones([160, 288, 288, 7], dtype=int)
        for i in range(28):
            print(i)
            if i < 10:
                train_tfr = 'gs://oai-challenge-us/tfrecords/valid/00' + str(i) + '-of-027.tfrecords'
            else:
                train_tfr = 'gs://oai-challenge-us/tfrecords/valid/0' + str(i) + '-of-027.tfrecords'
            raw_train_dataset = tf.data.TFRecordDataset(train_tfr)
            train_dataset = raw_train_dataset.map(parse_tf_img)
            j = 0
            for image, mask in train_dataset:
                train_ds[j, :, :, :] = image.numpy()
                mask_ds[j, :, :, :] = mask.numpy()
                j = j + 1
            train_dsds[i * 160:((i + 1) * 160), :, :, :] = train_ds
            mask_dsds[i * 160:((i + 1) * 160), :, :, :] = mask_ds
    if pre_train == True:
        return train_dsds
    else:
        return train_dsds, mask_dsds


AUTO = tf.data.AUTOTUNE
test_img, test_mask = load_data(data="Valid", pre_train=False)
# print('test_img',test_img.shape)
# print('test_mask',test_mask.shape)
# test_img = tf.convert_to_tensor(test_img)
# test_img = tf.cast(test_img, tf.float32)
# test_img = tf.data.Dataset.from_tensor_slices(test_img)
# test_img = test_img.batch(100).prefetch(AUTO)
# print('test_img',test_img)

# imgs_mask_test = mae_seg.predict(test_img, verbose=1)

# for i in range(100):
#     fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(15, 5))
#     ax[0, 0].imshow(test_img[i,:,:,:])
#     ax[1, 0].imshow(test_img[i,:,:,:])
#     ax[0, 1].imshow(test_mask[i,:,:,0])
#     ax[1, 1].imshow(imgs_mask_test[i,:,:,0])
#     ax[0, 2].imshow(test_mask[i,:,:,1])
#     ax[1, 2].imshow(imgs_mask_test[i,:,:,1])
#     ax[0, 3].imshow(test_mask[i,:,:,2])
#     ax[1, 3].imshow(imgs_mask_test[i,:,:,2])
#     ax[0, 4].imshow(test_mask[i,:,:,3])
#     ax[1, 4].imshow(imgs_mask_test[i,:,:,3])
#     ax[0, 5].imshow(test_mask[i,:,:,4])
#     ax[1, 5].imshow(imgs_mask_test[i,:,:,4])
#     ax[0, 6].imshow(test_mask[i,:,:,5])
#     ax[1, 6].imshow(imgs_mask_test[i,:,:,5])
#     ax[0, 7].imshow(test_mask[i,:,:,6])
#     ax[1, 7].imshow(imgs_mask_test[i,:,:,6])
#     address = "/home/caozi/Result/" + str(i) + ".jpg"
#     plt.savefig(address)
#     plt.close(fig)


print('valid_ds', valid_ds)
imgs_mask_test = mae_seg.predict(valid_ds, verbose=1, steps=validation_steps)
print('imgs_mask_test', imgs_mask_test.shape)
# for i in range(160):
#     fig, ax = plt.subplots(nrows=2, ncols=7, figsize=(15, 5))
#     ax[0, 0].imshow(test_img[i,:,:,:], cmap = 'gray')
#     ax[1, 0].imshow(test_img[i,:,:,:], cmap = 'gray')
#     ax[0, 1].imshow(test_mask[i,:,:,1], cmap = 'gray')
#     ax[1, 1].imshow(imgs_mask_test[i,:,:,1], cmap = 'gray')
#     ax[0, 2].imshow(test_mask[i,:,:,2], cmap = 'gray')
#     ax[1, 2].imshow(imgs_mask_test[i,:,:,2], cmap = 'gray')
#     ax[0, 3].imshow(test_mask[i,:,:,3], cmap = 'gray')
#     ax[1, 3].imshow(imgs_mask_test[i,:,:,3], cmap = 'gray')
#     ax[0, 4].imshow(test_mask[i,:,:,4], cmap = 'gray')
#     ax[1, 4].imshow(imgs_mask_test[i,:,:,4], cmap = 'gray')
#     ax[0, 5].imshow(test_mask[i,:,:,5], cmap = 'gray')
#     ax[1, 5].imshow(imgs_mask_test[i,:,:,5], cmap = 'gray')
#     ax[0, 6].imshow(test_mask[i,:,:,6], cmap = 'gray')
#     ax[1, 6].imshow(imgs_mask_test[i,:,:,6], cmap = 'gray')
#     # ax[0, 7].imshow(test_mask[i,:,:,0], vmin = 0, vmax = 0.005, cmap = 'gray')
#     # ax[1, 7].imshow(imgs_mask_test[i,:,:,0], vmin = 0, vmax = 0.005, cmap = 'gray')
#     address = "/home/caozi/Result/" + str(i) + ".jpg"
#     plt.savefig(address)
#     plt.close(fig)


# imgs_mask_test = mae_seg.predict(valid_ds, verbose=1, steps=validation_steps)
# print('imgs_mask_test', imgs_mask_test.shape)
# for i in range(100):
#     fig, ax = plt.subplots(figsize=(15, 5))
#     plt.imshow(imgs_mask_test[i,:,:,:])
#     address = "/home/caozi/Result/" + str(i) + ".jpg"
#     plt.savefig(address)
#     plt.close(fig)

dice = dice_coef_eval(imgs_mask_test, test_mask)
print(dice)
