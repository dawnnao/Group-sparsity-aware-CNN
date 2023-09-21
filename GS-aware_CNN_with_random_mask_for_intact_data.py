"""
Group sparsity-aware convolutional neural network
-------------------------------------------------
This Python file contains the implementation of group sparsity-aware CNN for
continuous missing data recovery of structural health monitoring as described in the paper

"Group sparsity-aware convolutional neural network for continuous missing data recovery of structural health monitoring"
by Zhiyi Tang, Yuequan Bao, and Hui Li.

Packages dependencies are listed in requirements.txt. The GS-aware CNN, and the data pre- and post-processing are packed into
the main function "gsn". It works as an automatic workflow once given required parameters and the data-to-recover.
Two example data sets have been included in folder \simulation_El_Centro and \simulation_impulse. Run this file with the
default parameters to check the examples.

Created on April 12 2020
author: Zhiyi Tang
email:  tang@stu.hit.edu.cn
web:    zhiyitang.info
------------------------------------------------------------------------------------------------------------------------
MIT License
Copyright <2020> <Zhiyi Tang, Yuequan Bao, and Hui Li>
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import tensorflow as tf
from keras.layers import Input, Multiply, Add, Subtract, Conv2D, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras import backend as K
from tensorflow.keras.utils import plot_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, Callback
import os
# os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # Windows
import hdf5storage
import math
from scipy.linalg import dft
import pickle
import time
from skimage.util.shape import view_as_windows
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#%% Sub-functions and main function gsn


# Make abbreviation
def abbr(vec):
    line = np.full([np.max(vec) + 2], np.nan)
    line[vec] = vec
    edge = []
    if not np.isnan(line[0]):
        edge.append(0)
    for n in np.arange(1, len(line)):
        if not np.isnan(line[n]) and np.isnan(line[n - 1]):
            edge.append(n)
        elif np.isnan(line[n]) and not np.isnan(line[n - 1]):
            edge.append(n - 1)
    edge = np.array(edge).reshape(np.int(len(edge) / 2), 2).T
    return edge


# Make abbreviated strings for file names
def tidy_name(edge):
    name_str = []
    for n in np.arange(0, edge.shape[1]):
        if edge[1, n] == edge[0, n]:
            name_str.append('%d' % edge[0, n])
        elif edge[1, n] == edge[0, n] + 1:
            name_str.append('%d,%d' % (edge[0, n], edge[1, n]))
        elif edge[1, n] > edge[0, n] + 1:
            name_str.append('%d-%d' % (edge[0, n], edge[1, n]))
    name_str = '_'.join(name_str)
    return name_str


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# Define unit for ylabel
def data_unit(data_name):
    return {
        'DPM': 'Displ. (mm)',
        'GPS': 'Displ. (mm)',
        'HPT': 'Displ. (mm)',
        'RHS': 'Tempr. (degree Celsius)',
        'TLT': 'Tilt (degree)',
        'UAN': 'Velocity (m/s)',
        'ULT': ' ',
        'VIB': 'Accel. (gal)',
        'VIC': 'Accel. (gal)',
        'VIF': 'Accel. (gal)',
        'STR': 'Strain',
        'U1': 'Velocity (mm/s)',
        'U2': 'Velocity (mm/s)',
        'U3': 'Velocity (mm/s)',
        'U4': 'Velocity (mm/s)',
        'U5': 'Velocity (mm/s)',
        'U6': 'Velocity (mm/s)',
        'U7': 'Velocity (mm/s)',
        'U8': 'Velocity (mm/s)',
        'U9': 'Velocity (mm/s)',
        'vel': 'Velocity (m/s)'
    }.get(data_name, ' ')  # default if x not found


# Fourier matrix
def dft2d(n):
    m = dft(n)
    return m


# Harmonic wavelet matrix:
def harmonic_wavelet(length, m, n):
    x = np.zeros((length, length), dtype='complex')
    interval = n - m
    piece_num = length / interval
    c = 0
    if piece_num % 1 != 0:
        raise Exception('Given length can not be exact divided by n-m.')
    else:
        piece_num = int(piece_num)
        for k in range(piece_num):
            w_raw = np.zeros(length, dtype='complex')
            w_raw[k * interval: (k + 1) * interval] = 1 + 0j
            x_raw = np.fft.ifft(w_raw)
            x_raw_normalized = x_raw / np.max(np.abs(x_raw))
            for p in range(interval):
                x_mid = np.roll(x_raw_normalized, p * piece_num)
                x[:, c] = x_mid
                c += 1
    return x


# Define sampling frequency
def data_fs(name):
    return {
        'DPM': 100,
        'GPS': 1,
        'HPT': 1,
        'RHS': 10,
        'TLT': 1,
        'UAN': 10,
        'ULT': 1,
        'VIB': 50,
        'VIC': 50,
        'VIF': 40,
        'RSG': 20,
        'vel': 100
    }.get(name, 1)  # 1 is default if x not found


# Main function
def gsn(data_path, data_name, tail, fs, duration, overlap, channel, sample_ratio, result_path, randseed,
        bad_channel=None, bad_sample_ratio=None, packet=1, regularizer_weight=1.0, batch_size=128, epochs=1200,
        harmonic_wavelet_interval=16, loss_weight_rate=32):

    #%% 1 Load data and prepare file and folder names

    data_raw = hdf5storage.loadmat(data_path + data_name + tail)
    data_all = data_raw[data_name]
    del data_raw

    if channel == 'all':
        channel = np.arange(0, data_all.shape[1])

    channel_num = len(channel)
    channel_str_abbr = tidy_name(abbr(channel))

    if bad_channel is None:

        result_folder = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                        '%.2f' % sample_ratio + '_' + str(randseed) + \
                        '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                        '_hw_interval_' + str(hw_interval) + \
                        '_lw_rate_' + '%03d' % loss_weight_rate + '/'

        result_file = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                      '%.2f' % sample_ratio + '_' + str(randseed) + \
                      '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                      '_hw_interval_' + str(hw_interval) + \
                      '_lw_rate_' + '%03d' % loss_weight_rate
    else:
        bad_channel_str = [str(i) for i in bad_channel]
        bad_channel_str_stack = '_'.join(bad_channel_str)
        bad_sample_ratio_str = ['%.2f' % i for i in bad_sample_ratio]
        bad_sample_ratio_str_stack = '_'.join(bad_sample_ratio_str)

        result_folder = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                        '%.2f' % sample_ratio + '_' + str(randseed) + \
                        '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                        '_hw_interval_' + str(hw_interval) + \
                        '_lw_rate_' + '%03d' % loss_weight_rate + \
                        '__[' + bad_channel_str_stack + ']_' + bad_sample_ratio_str_stack + '/'

        result_file = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                      '%.2f' % sample_ratio + '_' + str(randseed) + \
                      '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                      '_hw_interval_' + str(hw_interval) + \
                      '_lw_rate_' + '%03d' % loss_weight_rate + \
                      '__[' + bad_channel_str_stack + ']_' + bad_sample_ratio_str_stack

    create_folder(result_path + result_folder)
    channel_str = ['ch %d' % (i + 1) for i in channel]

    #%% 2 Preprocess data

    data_all = data_all[:, channel]
    data_split = view_as_windows(np.ascontiguousarray(data_all), (duration, data_all.shape[1]), duration - overlap)
    data_split = np.squeeze(data_split, axis=1)
    slice_num = data_split.shape[0]  # number of sections, split by duration
    print('Shape of data_split: ', data_split.shape)
    data_split_offset = np.nanmean(data_split, axis=1)
    data_split_norm = np.zeros(data_split_offset.shape)
    data_split_normalized = np.zeros(data_split.shape)

    for slice in range(slice_num):
        data_split_norm[slice] = np.nanmax(np.abs(data_split[slice] -
                                                data_split_offset[slice].reshape((1, channel_num))), axis=0)

        data_split_normalized[slice] = np.true_divide(data_split[slice] -
                                                      data_split_offset[slice].reshape((1, channel_num)),
                                                      data_split_norm[slice].reshape((1, channel_num)))

    data_hat = np.zeros(data_split.shape, dtype=np.complex128)
    data_masked_ignore_zero_all = np.zeros(data_split.shape)
    mask_matrix_all = np.zeros(data_split.shape)
    weights_complex = np.zeros(data_split.shape, dtype=np.complex128)
    recon_error_time_domain = np.zeros([slice_num, len(channel)])
    recon_error_freq_domain = np.zeros([slice_num, len(channel)])

    for slice in range(slice_num):
        start_time = time.time()
        data_time = data_split_normalized[slice]
        print('shape of data:\n', data_time.shape)
        dt = 1. / fs
        t = np.arange(0., duration / fs, dt)

        for f in range(len(channel)):
            fig = plt.figure(figsize=(18, 4))
            plt.plot(t, data_time[:, f])
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.legend([channel_str[f]], loc=1, fontsize=12)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel(data_unit(data_name), fontsize=12)
            plt.xlim(0, max(t))
            # plt.show()
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (channel[f] + 1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'signal_%d_slice-%02d.png' % (
            channel[f] + 1, slice + 1))
            plt.close()
            time.sleep(0.1)

        fig = plt.figure(figsize=(18, 4))
        plt.plot(t, data_time)
        # matplotlib.rcParams.update({'font.size': 12})
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        # matplotlib.rc('xtick', labelsize=12)
        # matplotlib.rc('ytick', labelsize=12)
        plt.legend(channel_str, loc=1, fontsize=12)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(data_unit(data_name), fontsize=12)
        plt.xlim(0, max(t))
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder + 'signal_all_slice-%02d.png' % (slice + 1))
        time.sleep(0.1)

        plt.close('all')

        #%% 3 Generate mask matrix

        # mask_matrix = ~np.isnan(data_time) * 1

        np.random.seed(randseed + slice)  # generate random seed

        if packet > 1:
            remain = np.mod(duration, packet)
            if bad_channel is None:
                mask_matrix_condensed = np.random.choice([0, 1],
                                                         size=(data_time.shape[0] // packet, data_time.shape[1]),
                                                         p=[1 - sample_ratio, sample_ratio])
                mask_matrix = np.vstack(
                    (np.repeat(mask_matrix_condensed, packet, axis=0), np.ones((remain, data_time.shape[1]))))
            else:
                mask_matrix_condensed = np.random.choice([0, 1],
                                                         size=(data_time.shape[0] // packet, data_time.shape[1]),
                                                         p=[1 - sample_ratio, sample_ratio])
                for b in range(len(bad_channel)):
                    mask_col = np.random.choice([0, 1], size=(data_time.shape[0] // packet),
                                                p=[1 - bad_sample_ratio[b], bad_sample_ratio[b]])
                    mask_matrix_condensed[:, bad_channel[b]] = mask_col
                mask_matrix = np.vstack(
                    (np.repeat(mask_matrix_condensed, packet, axis=0), np.ones((remain, data_time.shape[1]))))
        else:
            if bad_channel is None:
                mask_matrix = np.random.choice([0, 1], size=data_time.shape, p=[1 - sample_ratio, sample_ratio])
            else:
                mask_matrix = np.random.choice([0, 1], size=data_time.shape, p=[1 - sample_ratio, sample_ratio])
                for b in range(len(bad_channel)):
                    mask_col = np.random.choice([0, 1], size=(data_time.shape[0]),
                                                p=[1 - bad_sample_ratio[b], bad_sample_ratio[b]])
                    mask_matrix[:, bad_channel[b]] = mask_col

        mask_matrix_all[slice] = mask_matrix
        print('shape of mask_matrix:', mask_matrix.shape, '\n')

        #%% 4 Make masked data

        data_masked = np.nan_to_num(data_time) * mask_matrix

        #%% 5 Generate basis matrix

        basis_folder = 'basis_matrix/'
        basis_file = 'basis_' + str(duration) + '.npy'

        print('\nGenerating basis matrix...\n')
        m = 0
        n = hw_interval
        basis_matrix = harmonic_wavelet(duration, m, n)
        # basis_matrix = dft2d(duration)
        create_folder(result_path + basis_folder)
        print('\nShape of basis_matrix:\n', basis_matrix.shape, '\n')
        print('\nBasis_matrix:\n', basis_matrix, '\n')

        #%% 6 Construct neural network

        data_input1 = np.real(basis_matrix)
        data_input2 = np.imag(basis_matrix)
        data_input3 = mask_matrix
        data_output1 = np.real(data_masked.astype(complex))
        data_output2 = np.imag(data_masked.astype(complex))

        # Expand input dimension for CNN
        data_input1 = np.expand_dims(data_input1, 0)
        data_input1 = np.expand_dims(data_input1, -1)

        data_input2 = np.expand_dims(data_input2, 0)
        data_input2 = np.expand_dims(data_input2, -1)

        data_input3 = np.expand_dims(data_input3, 0)
        data_input3 = np.expand_dims(data_input3, -1)

        data_output1 = np.expand_dims(data_output1, 0)
        data_output1 = np.expand_dims(data_output1, -1)

        data_output2 = np.expand_dims(data_output2, 0)
        data_output2 = np.expand_dims(data_output2, -1)

        np.random.seed(randseed)
        tf.random.set_seed(randseed)

        class Regularizer(object):
            """Regularizer base class.
            """

            def __call__(self, x):
                return 0.

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        class GroupSparseRegularizer(Regularizer):

            def __init__(self, regularizer_weight):
                self.regularizer_weight = regularizer_weight

            def __call__(self, weight_matrix):
                return self.regularizer_weight * K.sum(tf.norm(weight_matrix,
                                                               ord='euclidean', axis=3))  # Frobenius norm

            def get_config(self):
                return {'regularizer_weight': float(self.regularizer_weight)}

        def model(data_input1, data_input2, data_input3, data_output1, data_output2, duration, num_channel):

            # input layer
            basis_real = Input(shape=(duration, duration, 1), name='Basis_real')
            basis_imag = Input(shape=(duration, duration, 1), name='Basis_imag')

            # Convolutional layer
            coeff_real = Conv2D(num_channel, kernel_size=(1, duration), activation='linear',
                                input_shape=(duration, duration, 1), name='Coeff_real', use_bias=False,
                                # kernel_regularizer=GroupSparseRegularizer(regularizer_weight=regularizer_weight))
                                kernel_regularizer=regularizers.l1(regularizer_weight))  # regularizers.l1(0.1)
            coeff_imag = Conv2D(num_channel, kernel_size=(1, duration), activation='linear',
                                input_shape=(duration, duration, 1), name='Coeff_imag', use_bias=False,
                                # kernel_regularizer=GroupSparseRegularizer(regularizer_weight=regularizer_weight))
                                kernel_regularizer=regularizers.l1(regularizer_weight))

            mask = Input(shape=(duration, num_channel, 1), name='Mask')

            # Feature maps as intermediate layer (need permute dimensions)
            item_ac = coeff_real(basis_real)
            item_ad = coeff_imag(basis_real)
            item_bc = coeff_real(basis_imag)
            item_bd = coeff_imag(basis_imag)

            item_ac_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_ac)
            item_ad_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_ad)
            item_bc_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_bc)
            item_bd_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_bd)

            # Recovery
            signal_real = Subtract(name='Signal_real')([item_ac_permuted, item_bd_permuted])
            signal_imag = Add(name='Signal_imag')([item_ad_permuted, item_bc_permuted])

            # masking
            signal_real_masked = Multiply(name='Sparse_signal_real')([signal_real, mask])
            signal_imag_masked = Multiply(name='Sparse_signal_imag')([signal_imag, mask])

            model = Model(inputs=[basis_real, basis_imag, mask], outputs=[signal_real_masked, signal_imag_masked])

            return model

        model = model(data_input1, data_input2, data_input3,
                      data_output1, data_output2,
                      duration, data_input3.shape[2])
        model.summary()

        if not os.path.exists(result_path + result_folder + 'model_v6.png'):
            plot_model(model, to_file='model_v6.png', show_shapes=True, show_layer_names=True)

        #%% 7 Solving

        opt = Adam(lr=0.0001)
        ##
        lw_real = K.variable(1.)
        lw_imag = K.variable(1.)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      loss_weights=[lw_real, lw_imag], metrics=['mae'])

        class LossWeightsScheduler(Callback):
            def __init__(self, lw_real, lw_imag, epochs, loss_weight_rate):
                self.lw_real = lw_real
                self.lw_imag = lw_imag
                self.epochs = epochs
                self.loss_weight_rate = loss_weight_rate

            def on_epoch_begin(self, epoch, logs={}):
                if epoch <= self.epochs:
                    K.set_value(self.lw_real, math.pow(2, (epoch / self.epochs * self.loss_weight_rate + 1)))
                    K.set_value(self.lw_imag, math.pow(2, (epoch / self.epochs * self.loss_weight_rate + 1)))

        class LossHistory(Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))

        loss_history = LossHistory()

        weights_folder = 'weights_history/'
        weights_slice_folder = 'slice_%02d/' % (slice + 1)
        create_folder(result_path + result_folder + weights_folder + weights_slice_folder)

        file_path_of_weights = result_path + result_folder + weights_folder + weights_slice_folder + \
                               'weights_slice-%02d-{epoch:03d}.hdf5' % (slice + 1)
        checkpoint = ModelCheckpoint(file_path_of_weights, monitor='loss', verbose=0, save_best_only=False,
                                     save_weights_only=True, mode='auto', period=50)

        history = model.fit(x=[data_input1, data_input2, data_input3], y=[data_output1, data_output2],
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[loss_history,
                                       LossWeightsScheduler(lw_real, lw_imag, epochs, loss_weight_rate),
                                       checkpoint])

        #%% 8 Check results

        # Plot training history
        print(history.history.keys())
        # print(len(loss_history.losses))

        # summarize history for loss
        fig = plt.figure(figsize=(6, 4))
        plt.semilogy(history.history['Sparse_signal_real_loss'], linestyle='-', color='red')
        plt.semilogy(history.history['Sparse_signal_imag_loss'], linestyle='--', color='blue')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        # plt.title('Model loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(['real part', 'imaginary part'], loc='upper right', fontsize=12)
        plt.tight_layout()
        plt.show()
        file_name = result_path + result_folder + 'training_history_slice-%02d' % (slice + 1)
        fig.savefig(file_name + '.png')
        fig.savefig(file_name + '.svg')
        fig.savefig(file_name + '.eps')
        fig.savefig(file_name + '.pdf')
        subprocess.call(
            'C:/Program Files/Inkscape/bin/inkscape.exe ' + file_name + '.svg ' '--export-emf=' + file_name + '.emf')
        time.sleep(0.1)

        # Get basis coeffs
        # weights_real = np.squeeze(model.get_weights()[0])
        weights_real = np.squeeze(model.get_weights()[0], axis=(0, 2))  # axis=(0, 3)
        print('shape of weights_real:', weights_real.shape, '\n')

        # weights_imag = np.squeeze(model.get_weights()[1])
        weights_imag = np.squeeze(model.get_weights()[1], axis=(0, 2))  #  axis=(0, 3)
        print('shape of weights_imag:', weights_imag.shape, '\n')

        weights_complex[slice] = weights_real + weights_imag * 1j
        print('shape of weights_complex:', weights_complex.shape, '\n')

        # Plot heatmap for the real part of coefficients
        fig = plt.figure(figsize=(3, 60))  # figsize=(6, 6)
        ax = sns.heatmap(weights_imag, cmap='coolwarm', square=True, xticklabels=1, yticklabels=128)
        sns.set(font_scale=1.0)
        # cbar_axes = ax.figure.axes[-1]
        # ax.figure.axes[-1].yaxis.label.set_size(12)
        # ax.tick_params(axis='both', which='major', labelsize=12)
        # ax.tick_params(axis='both', which='minor', labelsize=12)
        # plt.title('Imaginary part of basis coefficients')  # , fontsize=12
        # plt.xlabel('Column')
        # plt.ylabel('Row')
        # plt.xticks(np.arange(0, 2048, 32))
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder +
                    'basis_coeffs_signal_all_imag_slice-%02d_heatmap.pdf' % (slice + 1))
        plt.close()
        sns.reset_orig()

        # Plot heatmap for the imaginary part of coefficients for all channels
        fig = plt.figure(figsize=(3, 60))  # figsize=(6, 6)
        ax = sns.heatmap(weights_real, cmap='coolwarm', square=True, xticklabels=1, yticklabels=128)
        # sns.set(font_scale=0.05)
        # cbar_axes = ax.figure.axes[-1]
        # ax.figure.axes[-1].yaxis.label.set_size(12)
        # ax.tick_params(axis='both', which='major', labelsize=12)
        # ax.tick_params(axis='both', which='minor', labelsize=12)
        # plt.title('Real part of basis coefficients')  # , fontsize=12
        # plt.xlabel('Column')
        # plt.ylabel('Row')
        # plt.xticks(np.arange(0, 2048, 32))
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder +
                    'basis_coeffs_signal_all_real_slice-%02d_heatmap.pdf' % (slice + 1))
        plt.close()
        sns.reset_orig()

        # Plot real-part coefficients for all channels
        fig = plt.figure(figsize=(17, 4))
        plt.plot(weights_real)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        plt.title('Real part', fontsize=12)
        plt.xlabel('Point', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend(channel_str, loc=9, ncol=len(channel_str), fontsize=12)
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder + 'basis_coeffs_signal_all_real_slice-%02d.png' % (slice + 1))
        time.sleep(0.1)

        # Plot imaginary-part coefficients for all channels
        fig = plt.figure(figsize=(17, 4))
        plt.plot(weights_imag)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        plt.title('Imaginary part', fontsize=12)
        plt.xlabel('Point', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend(channel_str, loc=9, ncol=len(channel_str), fontsize=12)
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder + 'basis_coeffs_signal_all_imag_slice-%02d.png' % (slice + 1))
        time.sleep(0.1)

        # Plot coefficients of each channel respectively
        for f in range(len(channel)):
            fig = plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(weights_real[:, f])
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Real part', fontsize=12)
            plt.legend([channel_str[f]], loc=1, fontsize=12)
            plt.xlabel('Point', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.xlim(0, duration)

            plt.subplot(2, 1, 2)
            plt.plot(weights_imag[:, f])
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Imaginary part', fontsize=12)
            plt.legend([channel_str[f]], loc=1, fontsize=12)
            plt.xlabel('Point', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.xlim(0, duration)
            # plt.show()
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (channel[f] + 1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'basis_coeffs_signal_%d_slice-%02d.png' % (
            channel[f] + 1, slice + 1))
            plt.close()
            time.sleep(0.1)

        #%% Check reconstructed data

        data_hat_time = np.matmul(basis_matrix, weights_complex[slice])
        data_hat_time = data_hat_time * data_split_norm[slice] + data_split_offset[slice]
        data_time = data_time * data_split_norm[slice] + data_split_offset[slice]
        data_masked = data_masked * data_split_norm[slice] + data_split_offset[slice]

        data_hat[slice] = data_hat_time
        print('shape of data_hat_time:', data_hat_time.shape)

        recon_error_time_domain[slice] = np.linalg.norm((np.real(data_hat_time - data_time)), ord=2, axis=0) / \
                                         np.linalg.norm(np.real(data_time), ord=2, axis=0)
        print('\nrecon_error_time_domain:', recon_error_time_domain[slice])


        #%% Time domain:

        data_masked_ignore_zero = np.array(data_masked)
        data_masked_ignore_zero[data_masked_ignore_zero == 0] = np.nan  # data ignored zero for plotting
        data_masked_ignore_zero_all[slice] = data_masked_ignore_zero

        for f in channel:
            print('\nPlotting signal ' + str(f) + '\n')
            fig = plt.figure(figsize=(17, 4))
            plot_origin = plt.plot(t, np.real(data_time[:, list(channel).index(f)]), 'red')  # xkcd:lightish green
            plot_recover = plt.plot(t, np.real(data_hat_time[:, list(channel).index(f)]), 'blue')

            # plot background
            p = 0
            while p < duration:
                if mask_matrix[p, list(channel).index(f)] == 0:
                    plot_section = plt.axvspan(p / fs, (p + packet) / fs, facecolor='red', alpha=0.08)
                    p = p + packet
                else:
                    p = p + 1

            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Signal %s - Slice %03d - Comparison' % (f + 1, slice + 1), fontsize=12)
            plt.xlabel('Time (sec)', fontsize=12)
            plt.ylabel(data_unit(data_name), fontsize=12)
            plt.legend(['Original', 'Recovered'], fontsize=12)
            plt.xlim(0, max(t))
            plt.grid(True)
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (f + 1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_time_signal_' + str(
                f + 1) + '_slice-%03d.png' % (slice + 1))
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_time_signal_' + str(
                f + 1) + '_slice-%03d.pdf' % (slice + 1))
            time.sleep(0.1)
            plt.close('all')

        #%% Frequency domain:

        data_freq = np.fft.fft(np.real(data_time), axis=0)
        data_hat_freq = np.fft.fft(np.real(data_hat_time), axis=0)

        recon_error_freq_domain[slice] = np.linalg.norm((data_hat_freq - data_freq), ord=2, axis=0) / \
                                         np.linalg.norm(data_freq, ord=2, axis=0)

        x_axis_freq = np.arange(0, fs, fs / duration)
        for f in channel:
            print('\nPlotting signal ' + str(f) + '\n')
            fig = plt.figure(figsize=(17, 4))
            plt.plot(x_axis_freq, np.abs(data_freq[:, list(channel).index(f)]), 'red')  # xkcd:lightish green
            plt.plot(x_axis_freq, np.abs(data_hat_freq[:, list(channel).index(f)]), 'blue')
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Signal %s - Slice %03d - Comparison' % (f + 1, slice + 1), fontsize=12)
            plt.xlabel('Frequency (Hz)', fontsize=12)
            plt.ylabel(data_unit(data_name), fontsize=12)
            plt.legend(['Original', 'Recovered'], fontsize=12)
            plt.xlim(0, fs / 2 + 1)
            plt.grid(True)
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (f + 1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_freq_signal_' + str(
                f + 1) + '_slice-%03d.png' % (slice + 1))
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_freq_signal_' + str(
                f + 1) + '_slice-%03d.pdf' % (slice + 1))
            time.sleep(0.1)
            plt.close('all')

        elapsed_time = time.time() - start_time
        print(time.strftime('Elapsed time: %H:%M:%S\n', time.gmtime(elapsed_time)))

    #%% Combine segments

    if overlap == 0:
        linked_data_time = data_split[:, :, :].reshape(-1, data_split.shape[2])

        linked_data_hat_time = data_hat[:, :, :].reshape(-1, data_hat.shape[2])

        linked_data_masked_ignore_zero_all = data_masked_ignore_zero_all[:, :, :].reshape(-1, data_masked_ignore_zero_all.shape[2])

        linked_mask_matrix_all = mask_matrix_all[:, :, :].reshape(-1, mask_matrix_all.shape[2])

    else:

        linked_data_time = data_split[:, int(overlap / 2):int(-overlap / 2), :].reshape(-1, data_split.shape[2])

        linked_data_hat_time = data_hat[:, int(overlap / 2):int(-overlap / 2), :].reshape(-1, data_hat.shape[2])

        linked_data_masked_ignore_zero_all = \
            data_masked_ignore_zero_all[:, int(overlap / 2):int(-overlap / 2), :].reshape(-1, data_masked_ignore_zero_all.shape[2])

        linked_mask_matrix_all = mask_matrix_all[:, int(overlap / 2):int(-overlap / 2), :].reshape(-1, mask_matrix_all.shape[2])

    #%% 9 Save results

    results = {
        'data_split': data_split,
        'data_hat': data_hat,
        'data_masked_ignore_zero_all': data_masked_ignore_zero_all,
        'mask_matrix_all': mask_matrix_all,
        'weights_complex': weights_complex,
        'recon_error_time_domain': recon_error_time_domain,
        'recon_error_freq_domain': recon_error_freq_domain,
        'linked_data_time': linked_data_time,
        'linked_data_hat_time': linked_data_hat_time,
        'linked_data_masked_ignore_zero_all': linked_data_masked_ignore_zero_all,
        'linked_mask_matrix_all': linked_mask_matrix_all
    }

    # Collect metrics for parallel plot
    collect_results_folder = 'collect_results/'
    collect_results_file = 'collect_results'
    create_folder(result_path + collect_results_folder)

    recon_error_time_domain_mean_all = float(np.mean(recon_error_time_domain))

    with open(result_path + collect_results_folder + collect_results_file + '.txt', 'a+') as f:
        f.write('%5.2f, %03d, %03d, %02d, %04d, %7.5f\n' % (regularizer_weight, batch_size,
                                                            harmonic_wavelet_interval, loss_weight_rate,
                                                            epochs, recon_error_time_domain_mean_all))
    f.close()

    # Save metrics in txt locally
    np.savetxt(result_path + result_folder + result_file + '.txt', recon_error_time_domain * 100, fmt='%.2f',
               header='row: slice, column: channel')
    print('\nResults metrics saved in: ' + result_file + '.txt')

    # Save results file locally
    with open(result_path + result_folder + result_file + '.pickle', 'wb') as f:
        pickle.dump(results, f)
    f.close()
    print('\nResults saved in: ' + result_file + '.pickle')

    hdf5storage.savemat(result_path + result_folder + result_file + '.mat', results)
    print('\nResults saved in: ' + result_file + '.mat')

    K.clear_session()


#%% 0 User input

# A example
# data_path = './simulation_impulse/'
# data_name = ['VIB']  # DPM

# data_path = './earthquake_response_outrange/'
# data_name = ['vel']  # DPM

data_path = './dynamic_strain_of_ocean_platform/'
data_name = ['STR']

tail = '.mat'
channel = 'all'  # 'all' or a list. For example: [0, 1, 5]

sample_ratio = np.array([0.5])  # input sampling ratio here
packet = np.array([100])  # number of point in each missing segment

randseed = np.arange(0, 1)  # seeds for random number
regularizer_weight = np.array([10.0])
batch_size = 128
epochs = 120
harmonic_wavelet_interval = np.array([1])
loss_weight_rate = 24

#%%

for pac in packet:
    for d_name in data_name:
        fs = data_fs(d_name)
        duration = 8192  # no larger than the length of data-to-recover
        overlap = 0  # 512
        result_path = '%s/results/' % data_path

        for sratio in sample_ratio:
            for rseed in randseed:
                for rw in regularizer_weight:
                    for hw_interval in harmonic_wavelet_interval:
                        gsn(data_path=data_path, data_name=d_name, tail=tail, fs=fs, duration=duration, overlap=overlap,
                            channel=channel, sample_ratio=sratio, result_path=result_path, randseed=rseed, packet=pac,
                            regularizer_weight=rw, batch_size=batch_size,
                            epochs=epochs, harmonic_wavelet_interval=hw_interval, loss_weight_rate=loss_weight_rate)
                        print('\nSeed %d done.\n' % rseed)
                        time.sleep(1)
