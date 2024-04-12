import keras.utils
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model

inp = Input(shape=(240, 320, 1))

n_filters = 512
max_p = inp

for _ in range(3):
    conv = Conv2D(filters=n_filters, kernel_size=5, activation='relu')(max_p)
    max_p = MaxPooling2D(pool_size=(2, 2))(conv)

    n_filters //= 2

conv = Conv2D(filters=n_filters, kernel_size=3, activation='relu')(max_p)
max_p = MaxPooling2D(pool_size=(2, 2))(conv)

flatten = Flatten()(max_p)
dense = Dense(128, activation='relu')(flatten)

head_out = Dense(1, activation='linear', name='head_out')(dense)
neck_out = Dense(1, activation='linear', name='neck_out')(dense)
shoulder_to_shoulder_out = Dense(1, activation='linear', name='shoulder_to_shoulder_out')(dense)
shoulder_to_wrist_out = Dense(1, activation='linear', name='shoulder_to_wrist_out')(dense)
torso_out = Dense(1, activation='linear', name='torso_out')(dense)
bicep_out = Dense(1, activation='linear', name='bicep_out')(dense)
wrist_out = Dense(1, activation='linear', name='wrist_out')(dense)
chest_out = Dense(1, activation='linear', name='chest_out')(dense)
waist_out = Dense(1, activation='linear', name='waist_out')(dense)
pelvis_out = Dense(1, activation='linear', name='pelvis_out')(dense)
inner_leg_out = Dense(1, activation='linear', name='inner_leg_out')(dense)
thigh_out = Dense(1, activation='linear', name='thigh_out')(dense)
knee_out = Dense(1, activation='linear', name='knee_out')(dense)
calf_out = Dense(1, activation='linear', name='calf_out')(dense)

model = Model(
    inputs=inp,
    outputs=[
        head_out,
        neck_out,
        shoulder_to_shoulder_out,
        shoulder_to_wrist_out,
        torso_out,
        bicep_out,
        wrist_out,
        chest_out,
        waist_out,
        pelvis_out,
        inner_leg_out,
        thigh_out,
        knee_out,
        calf_out
    ],
    name='conv_bodies'
)
import os
import cv2
import random


class Database_Loader(keras.utils.Sequence):

    def __init__(self, image_location, data_location, sample_count, batch_size, dataset="Surreact", shuffle=True,
                 seed=0, input_dimensions=(240, 320), prefix="Avatar_", use_memory=False, load_data=True,
                 random_sample=False):
        self.image_location = image_location
        self.data_location = data_location
        self.sample_count = sample_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.input_dimensions = input_dimensions
        self.prefix = prefix
        self.use_memory = use_memory
        self.load_data = load_data
        self.random_sample = random_sample
        self.dataset = dataset

        if self.random_sample:
            self.IDs = random.sample(range(len(os.listdir(self.image_location))), self.sample_count)
        else:
            self.IDs = [x for x in range(self.sample_count)]
            self._load_data()

        super().__init__(workers=2, use_multiprocessing=True)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.sample_count / self.batch_size))

    def _load_data(self):
        if self.load_data:
            self.data = dict()
            for index in self.IDs:
                self.data[index] = np.load(os.path.join(self.data_location + self.prefix + f"{index:06d}.npy"))

        if self.use_memory:
            self.images = dict()
            for index in self.IDs:
                self.images[index] = cv2.imread(os.path.join(self.image_location + self.prefix + f"{index:06d}.png"),
                                                cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, index):
        X = np.empty(shape=(self.batch_size, 240, 320))
        y = {
            'head_out': [],
            'neck_out': [],
            'shoulder_to_shoulder_out': [],
            'shoulder_to_wrist_out': [],
            'torso_out': [],
            'bicep_out': [],
            'wrist_out': [],
            'chest_out': [],
            'waist_out': [],
            'pelvis_out': [],
            'inner_leg_out': [],
            'thigh_out': [],
            'knee_out': [],
            'calf_out': []
        }

        start_index = index * self.batch_size + 1

        for i in range(self.batch_size):
            name = f"{self.IDs[(start_index + i) % self.sample_count]:06d}"

            if self.load_data:
                current_measurement = self.data[self.IDs[(start_index + i) % self.sample_count]][:-1]
            else:
                current_measurement = np.load(os.path.join(self.data_location + self.prefix + name + ".npy"))[:-1]

            if self.use_memory:
                X[i,] = self.images[self.IDs[(start_index + i) % self.sample_count]]

            else:
                X[i,] = cv2.imread(os.path.join(self.image_location + self.prefix + name + ".png"),
                                   cv2.IMREAD_GRAYSCALE)

            if self.dataset == "Surreact":
                y['head_out'].append([current_measurement[0]])
                y['neck_out'].append([current_measurement[1]])
                y['shoulder_to_shoulder_out'].append([current_measurement[2]])
                y['shoulder_to_wrist_out'].append([current_measurement[4]])
                y['torso_out'].append([current_measurement[5]])
                y['bicep_out'].append([current_measurement[6]])
                y['wrist_out'].append([current_measurement[7]])
                y['chest_out'].append([current_measurement[8]])
                y['waist_out'].append([current_measurement[9]])
                y['pelvis_out'].append([current_measurement[10]])
                y['inner_leg_out'].append([current_measurement[12]])
                y['thigh_out'].append([current_measurement[13]])
                y['knee_out'].append([current_measurement[14]])
                y['calf_out'].append([current_measurement[15]])

        for key, value in y.items():
            y[key] = np.array(value)

        return X, y

    def on_epoch_end(self):
        if self.random_sample:
            self.IDs = random.sample(range(len(os.listdir(self.image_location))), self.sample_count)
            self._load_data()

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.IDs)
        else:
            self.IDs = np.arange(self.sample_count)


quickTrain = {
    'image_location': "../Surreact-APose/train/imgs_nobg_frontEDITED/",
    'data_location': "../Surreact-APose/train/measurements/",
    'sample_count': 30000,
    'batch_size': 11,
    'seed': 69,  # np.random.randint(0, 10000),
    'use_memory': False,
    'random_sample': True,
    'shuffle': True}

quickValidate = {
    'image_location': "../Surreact-APose/train/imgs_nobg_frontEDITED/",
    'data_location': "../Surreact-APose/train/measurements/",
    'sample_count': 5000,
    'batch_size': 11,
    'seed': 69,  # np.random.randint(0, 10000),
    'use_memory': False,
    'random_sample': False}

realDataTest = {
    'image_location': "../bodym-dataset/front/images/",
    'data_location': "../bodym-dataset/front/measurements/",
    'sample_count': 8978,
    'batch_size': 4,
    'use_memory': False,
    'random_sample': False,
    'shuffle': False}

train_generator = Database_Loader(**quickTrain)
validation_generator = Database_Loader(**quickValidate)

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.losses import MeanSquaredError
from keras_tqdm import TQDMCallback

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=.001,
    decay_steps=30000,
    decay_rate=0.96,
    staircase=True)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss={
        'head_out': MeanSquaredError(),
        'neck_out': MeanSquaredError(),
        'shoulder_to_shoulder_out': MeanSquaredError(),
        'shoulder_to_wrist_out': MeanSquaredError(),
        'torso_out': MeanSquaredError(),
        'bicep_out': MeanSquaredError(),
        'wrist_out': MeanSquaredError(),
        'chest_out': MeanSquaredError(),
        'waist_out': MeanSquaredError(),
        'pelvis_out': MeanSquaredError(),
        'inner_leg_out': MeanSquaredError(),
        'thigh_out': MeanSquaredError(),
        'knee_out': MeanSquaredError(),
        'calf_out': MeanSquaredError()
    },
    # loss_weights={
    #     'head_out': 2.0,
    #     'neck_out': 1.0,
    #     'shoulder_to_shoulder_out': 2.0,
    #     'arm_out': 1.0,
    #     'shoulder_to_wrist_out': 1.0,
    #     'torso_out': 1.0,
    #     'bicep_out': 1.0,
    #     'wrist_out': 1.0,
    #     'chest_out': 1.0,
    #     'waist_out': 1.0,
    #     'pelvis_out': 1.0,
    #     'leg_out': 1.0,
    #     'inner_leg_out': 3.0,
    #     'thigh_out': 1.0,
    #     'knee_out': 1.0,
    #     'calf_out': 1.0
    # },
    # loss='mse',
    metrics={
        'head_out': ['mae'],
        'neck_out': ['mae'],
        'shoulder_to_shoulder_out': ['mae'],
        'shoulder_to_wrist_out': ['mae'],
        'torso_out': ['mae'],
        'bicep_out': ['mae'],
        'wrist_out': ['mae'],
        'chest_out': ['mae'],
        'waist_out': ['mae'],
        'pelvis_out': ['mae'],
        'inner_leg_out': ['mae'],
        'thigh_out': ['mae'],
        'knee_out': ['mae'],
        'calf_out': ['mae']
    },
)
keras.utils.set_random_seed(42)

checkpoint_filepath = './models/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_best_only=True)

model.fit(
    x=train_generator,
    validation_data=validation_generator,
    callbacks=[

        TensorBoard(write_graph=False, log_dir="./logs"),
        TQDMCallback(),
        model_checkpoint_callback,
    ],
    batch_size=11,
    epochs=30,
    verbose=0
)

model.save("./models/full/model.keras")
