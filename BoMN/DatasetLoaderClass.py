import numpy as np
from keras.utils import Sequence
import os
import cv2


class DatasetLoader(Sequence):

    def __init__(self, image_location, data_location, samples, batch_size, dataset, shuffle=True, seed=0,
                 input_dimensions=(240, 320), prefix="Avatar_", side=False, height=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.input_dimensions = input_dimensions
        self.prefix = prefix
        self.side = side
        self.height = height

        super().__init__(workers=5, use_multiprocessing=True)
        self.IDs = []
        self._load_data(samples, data_location, image_location, dataset)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.IDs) / self.batch_size))

    def _load_data(self, samples, data_locations, image_locations, datasets):
        self.IDs = []
        self.data = dict()
        id_counter = 0
        for data_source_index in range(len(image_locations)):
            source_image_count = samples[data_source_index]
            source_data_location = data_locations[data_source_index]
            source_image_location = image_locations[data_source_index]
            source_dataset = datasets[data_source_index]

            for index in range(source_image_count):
                self.IDs.append(id_counter)
                self.data[id_counter] = [source_image_location,
                                         source_dataset,
                                         index,
                                         np.load(os.path.join(source_data_location + self.prefix + f"{index:06d}.npy"))]
                id_counter += 1

    def __getitem__(self, index):
        X = np.empty(shape=(self.batch_size, 320, 240))
        if self.side:
            X = np.empty(shape=(self.batch_size, 320, 240, 2))
        y = {
            'chest_out': [],
            'waist_out': [],
            'pelvis_out': [],
            'bicep_out': [],
            'thigh_out': [],
            'shoulder_to_wrist_out': [],
            'leg_out': [],
            'calf_out': [],
            'wrist_out': [],
            'shoulder_to_shoulder_out': [],
        }

        height = []
        start_index = index * self.batch_size + 1
        for i in range(self.batch_size):
            sample_ID = self.IDs[(start_index + i)]
            sample_properties = self.data[sample_ID]
            current_measurement = sample_properties[3]
            current_filename = f"{self.prefix}{sample_properties[2]:06d}.png"

            X[i, :, :, 0] = cv2.imread(os.path.join(sample_properties[0] + "images_front/" + current_filename), cv2.IMREAD_GRAYSCALE)
            if self.side:
                X[i, :, :, 1] = cv2.imread(os.path.join(sample_properties[0] + "images_side/" + current_filename), cv2.IMREAD_GRAYSCALE)

            if sample_properties[1] == "Surreact":
                y['chest_out'].append([current_measurement[0]])
                y['waist_out'].append([current_measurement[1]])
                y['pelvis_out'].append([current_measurement[2]])
                y['bicep_out'].append([current_measurement[4]])
                y['thigh_out'].append([current_measurement[5]])
                y['shoulder_to_wrist_out'].append([current_measurement[7]])
                y['leg_out'].append([current_measurement[8]])
                y['calf_out'].append([current_measurement[9]])
                y['wrist_out'].append([current_measurement[11]])
                y['shoulder_to_shoulder_out'].append([current_measurement[13]])
                if self.height:
                    height.append(current_measurement[16])

            if sample_properties[1] == "BodyM":
                y['chest_out'].append([current_measurement[4]])
                y['waist_out'].append([current_measurement[12]])
                y['pelvis_out'].append([current_measurement[7]])
                y['bicep_out'].append([current_measurement[2]])
                y['thigh_out'].append([current_measurement[11]])
                y['shoulder_to_wrist_out'].append([current_measurement[1]])
                y['leg_out'].append([current_measurement[8]])
                y['calf_out'].append([current_measurement[3]])
                y['wrist_out'].append([current_measurement[13]])
                y['shoulder_to_shoulder_out'].append([current_measurement[9]])
                if self.height:
                    height.append(current_measurement[6])

        for key, value in y.items():
            y[key] = np.array(value)

        if self.height:
            height = np.array(height)
            return [X, height], y

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.IDs)
        else:
            self.IDs = np.arange(len(self.IDs))
