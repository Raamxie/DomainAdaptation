import os
import random
import cv2
import keras.utils
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.losses import MeanSquaredError
from keras.models import Model
from keras.optimizers import Adam

keras.utils.set_random_seed(42)
np.random.seed(69)
random.seed(69)

#TRAINING VARIABLES

sample_count = 79999
batch_size = 5
epoch_count = 10
validation_count = 20000
initial_learning_rate = .001
decay_steps = sample_count//batch_size
decay_rate = 0.97
dataset_name = "SuToBoCM"

folder_path = f"{dataset_name} - {sample_count}s{epoch_count}e{validation_count}v {'' if initial_learning_rate == 0.001 else initial_learning_rate + 'ilr'}{decay_steps}s{decay_rate}d"


#ARCHITECTURE

inp = Input(shape=(320, 240, 1))

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

chest_out = Dense(1, activation='linear', name='chest_out')(dense)
waist_out = Dense(1, activation='linear', name='waist_out')(dense)
pelvis_out = Dense(1, activation='linear', name='pelvis_out')(dense)
# neck_out = Dense(1, activation='linear', name='neck_out')(dense)
bicep_out = Dense(1, activation='linear', name='bicep_out')(dense)
thigh_out = Dense(1, activation='linear', name='thigh_out')(dense)
# knee_out = Dense(1, activation='linear', name='knee_out')(dense)
shoulder_to_wrist_out = Dense(1, activation='linear', name='shoulder_to_wrist_out')(dense)
leg_out = Dense(1, activation='linear', name='leg_out')(dense)
calf_out = Dense(1, activation='linear', name='calf_out')(dense)
# head_out = Dense(1, activation='linear', name='head_out')(dense)
wrist_out = Dense(1, activation='linear', name='wrist_out')(dense)
#arm_out = Dense(1, activation='linear', name='arm_out')(dense)
shoulder_to_shoulder_out = Dense(1, activation='linear', name='shoulder_to_shoulder_out')(dense)
# torso_out = Dense(1, activation='linear', name='torso_out')(dense)
#inner_leg_out = Dense(1, activation='linear', name='inner_leg_out')(dense)

model = Model(
    inputs=inp,
    outputs=[
        chest_out,
        waist_out,
        pelvis_out,
        # neck_out,
        bicep_out,
        thigh_out,
        # knee_out,
        shoulder_to_wrist_out,
        leg_out,
        calf_out,
        # head_out,
        wrist_out,
        #arm_out,
        shoulder_to_shoulder_out,
        # torso_out,
        #inner_leg_out
    ],
    name='conv_bodies'
)



#DATA LOADING
class DatasetLoader(keras.utils.Sequence):

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

        super().__init__(workers=2, use_multiprocessing=True)

        if self.random_sample:
            self.IDs = random.sample(range(len(os.listdir(self.image_location))), self.sample_count)
        else:
            self.IDs = [x for x in range(self.sample_count)]
            self._load_data()

        super().__init__(workers=1, use_multiprocessing=False)
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
        X = np.empty(shape=(self.batch_size, 320, 240))
        y = {
            'chest_out': [],
            'waist_out': [],
            'pelvis_out': [],
            # 'neck_out': [],
            'bicep_out': [],
            'thigh_out': [],
            # 'knee_out': [],
            'shoulder_to_wrist_out': [],
            'leg_out': [],
            'calf_out': [],
            # 'head_out': [],
            'wrist_out': [],
            'arm_out': [],
            'shoulder_to_shoulder_out': [],
            # 'torso_out': [],
            # 'inner_leg_out': [],
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
                y['chest_out'].append([current_measurement[0]])
                y['waist_out'].append([current_measurement[1]])
                y['pelvis_out'].append([current_measurement[2]])
                # y['neck_out'].append([current_measurement[3]])
                y['bicep_out'].append([current_measurement[4]])
                y['thigh_out'].append([current_measurement[5]])
                # y['knee_out'].append([current_measurement[6]])
                y['shoulder_to_wrist_out'].append([current_measurement[7]])
                y['leg_out'].append([current_measurement[8]])
                y['calf_out'].append([current_measurement[9]])
                # y['head_out'].append([current_measurement[10]])
                y['wrist_out'].append([current_measurement[11]])
                y['arm_out'].append([current_measurement[12]])
                y['shoulder_to_shoulder_out'].append([current_measurement[13]])
                # y['torso_out'].append([current_measurement[14]])
                # y['inner_leg_out'].append([current_measurement[15]])

        for key, value in y.items():
            y[key] = np.array(value)

        return X, y

    def on_epoch_end(self):
        if self.random_sample:
            self.IDs = random.sample(range(len(os.listdir(self.image_location))), self.sample_count)
            self._load_data()

        if self.shuffle:
            np.random.shuffle(self.IDs)
        else:
            self.IDs = np.arange(self.sample_count)


quickTrain = {
    'image_location': "Export/Surreact-APose/train/images_front/",
    'data_location': "Export/Surreact-APose/train/measurements/",
    'sample_count': sample_count,
    'batch_size': batch_size,
    'seed': 69,  # np.random.randint(0, 10000),
    'use_memory': False,
    'random_sample': False,
    'shuffle': True}

quickValidate = {
    'image_location': "Export/Surreact-APose/test/images_front/",
    'data_location': "Export/Surreact-APose/test/measurements/",
    'sample_count': validation_count,
    'batch_size': batch_size,
    'seed': 69,  # np.random.randint(0, 10000),
    'use_memory': False,
    'random_sample': False}


train_generator = DatasetLoader(**quickTrain)
validation_generator = DatasetLoader(**quickValidate)


#SCHEDULING
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)


#COMPILATION AND FIT
model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss={
        'chest_out': MeanSquaredError(),
        'waist_out': MeanSquaredError(),
        'pelvis_out': MeanSquaredError(),
        # 'neck_out': MeanSquaredError(),
        'bicep_out': MeanSquaredError(),
        'thigh_out': MeanSquaredError(),
        # 'knee_out': MeanSquaredError(),
        'shoulder_to_wrist_out': MeanSquaredError(),
        'leg_out': MeanSquaredError(),
        'calf_out': MeanSquaredError(),
        # 'head_out': MeanSquaredError(),
        'wrist_out': MeanSquaredError(),
        #'arm_out': MeanSquaredError(),
        'shoulder_to_shoulder_out': MeanSquaredError(),
        # 'torso_out': MeanSquaredError(),
        # 'inner_leg_out': MeanSquaredError()
    },
    metrics={
        'chest_out': ['mae'],
        'waist_out': ['mae'],
        'pelvis_out': ['mae'],
        # 'neck_out': ['mae'],
        'bicep_out': ['mae'],
        'thigh_out': ['mae'],
        # 'knee_out': ['mae'],
        'shoulder_to_wrist_out': ['mae'],
        'leg_out': ['mae'],
        'calf_out': ['mae'],
        # 'head_out': ['mae'],
        'wrist_out': ['mae'],
        #'arm_out': ['mae'],
        'shoulder_to_shoulder_out': ['mae'],
        # 'torso_out': ['mae'],
        # 'inner_leg_out': ['mae'],
    },
)

checkpoint_filepath = './Test/test.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_best_only=True)

model.fit(
    x=train_generator,
    validation_data=validation_generator,
    callbacks=[

        TensorBoard(write_graph=True, log_dir="./Test/" + folder_path),
        model_checkpoint_callback,
    ],
    batch_size=batch_size,
    epochs=epoch_count,
    verbose=1
)