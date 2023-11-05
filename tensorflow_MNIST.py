import tensorflow as ts
import numpy
import tensorflow_datasets as tfds

# Load the dataset into training and testing sets
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Normalize the data
def normalize_img(img, label):
    return ts.cast(img, ts.float32) / 255., label

# Apply the normalization function to the dataset
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=ts.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(ts.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls= ts.data.experimental.AUTOTUNE
)

ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(ts.data.experimental.AUTOTUNE)

# Create the model
model = ts.keras.Sequential([
    ts.keras.layers.Flatten(input_shape=(28, 28)),
    ts.keras.layers.Dense(128, activation='relu'),
    ts.keras.layers.Dense(10)
])
model.compile(
    optimizer= ts.keras.optimizers.Adam(0.001),
    loss= ts.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[ts.keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test
)