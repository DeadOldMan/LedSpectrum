#!/usr/bin/env python
#
# Licensed under the BSD license.  See full license in LICENSE file.
#
# Author: Tomas Mueller
#
# Third party dependencies:
#
# pyaudio: for audio input/output - http://pyalsaaudio.sourceforge.net/
# numpy: for FFT calcuation - http://www.numpy.org/

import argparse
import csv
import logging
import sys

import pyaudio
import numpy as np
import bibliopixel.colors as colors

_MIN_FREQUENCY = float(50) # 50 Hz
_MAX_FREQUENCY = float(15000) # 15000 HZ
_MAX_BARS      = 12 #12 LED bars
_MAX_HEIGHT    = 20 #20 LEDs per BAR
_CHUNK_SIZE    = 2048  # Use a multiple of 8 
_SERVER_IP     = '192.168.210.203'

#-----------------------------------------------------------------------
#---- FFT start
#-----------------------------------------------------------------------
def piff(val, chunk_size, sample_rate):
    '''Return the power array index corresponding to a particular frequency.'''
    return int(chunk_size * val / sample_rate)

def calculate_levels(data, chunk_size, sample_rate, frequency_limits, channels, outbars):
    '''Calculate frequency response for each channel defined in frequency_limits

    Initial FFT code inspired from the code posted here:
    http://www.raspberrypi.org/phpBB3/viewtopic.php?t=35838&p=454041

    Optimizations from work by Scott Driscoll:
    http://www.instructables.com/id/Raspberry-Pi-Spectrum-Analyzer-with-RGB-LED-Strip-/
    '''

    # create a numpy array, taking just the left channel if stereo
    data_stereo = np.frombuffer(data, dtype=np.int16)
    if channels == 2:
        data = np.empty(len(data) / (2 * channels))  # data has 2 bytes per channel
        data[:] = data_stereo[::2]  # pull out the even values, just using left channel
    elif channels == 1:
        data = data_stereo

    # if you take an FFT of a chunk of audio, the edges will look like
    # super high frequency cutoffs. Applying a window tapers the edges
    # of each end of the chunk down to zero.
    window = np.hanning(len(data))
    data = data * window

    # Apply FFT - real data
    fourier = np.fft.rfft(data)

    # Remove last element in array to make it the same size as chunk_size
    fourier = np.delete(fourier, len(fourier) - 1)

    # Calculate the power spectrum
    power = np.abs(fourier) ** 2

    matrix = np.zeros(outbars)
    for i in range(outbars):
        # take the log10 of the resulting sum to approximate how human ears perceive sound levels
        matrix[i] = np.log10(np.sum(power[piff(frequency_limits[i][0], chunk_size, sample_rate)
                                          :piff(frequency_limits[i][1], chunk_size, sample_rate):1]))

    return matrix
#-----------------------------------------------------------------------
#---- FFT end
#-----------------------------------------------------------------------

def calculate_channel_frequency(min_frequency, max_frequency ):
    '''Calculate frequency values for each channel, taking into account custom settings.'''

    # How many channels do we need to calculate the frequency for
    logging.debug("Normal Channel Mapping is being used.")
    channel_length = _MAX_BARS

    logging.debug("Calculating frequencies for %d channels.", channel_length)
    octaves = (np.log(max_frequency / min_frequency)) / np.log(2)
    logging.debug("octaves in selected frequency range ... %s", octaves)
    octaves_per_channel = octaves / channel_length
    frequency_limits = []
    frequency_store = []

    frequency_limits.append(min_frequency)
    logging.debug("Custom channel frequencies are not being used")
    for i in range(1, _MAX_BARS + 1):
        frequency_limits.append(frequency_limits[-1]
                                * 10 ** (3 / (10 * (1 / octaves_per_channel))))
    for i in range(0, channel_length):
        frequency_store.append((frequency_limits[i], frequency_limits[i + 1]))
        logging.debug("channel %d is %6.2f to %6.2f ", i, frequency_limits[i],
                      frequency_limits[i + 1])
        print("channel %d is %6.2f to %6.2f " %( i, frequency_limits[i], frequency_limits[i + 1]))
 

    return frequency_store

columns = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
decay = .9
led_colors = [colors.hue_helper(y, _MAX_HEIGHT, 0) for y in range(_MAX_HEIGHT)]

def update_lights(matrix, mean, std):
    '''Update the state of all the lights based upon the current frequency response matrix'''
    
    led.all_off()
    
    for x in range(0, len(matrix) ):
        # normalize output
        height = matrix[x] - 9.0
        height = height / 5
        if height < .05:
            height = .05
        elif height > 1.0:
            height = 1.0

        if height < columns[x]:
            columns[x] = columns[x] * decay
            height = columns[x]
        else:
            columns[x] = height

        numPix = int(round(height*(_MAX_HEIGHT+1)))

        for y in range(_MAX_HEIGHT):
            if y < int(numPix):
                led.set(x, _MAX_HEIGHT - y - 1, led_colors[y])
        #led.drawLine(i, _MAX_HEIGHT - 1, i, _MAX_HEIGHT - int(numPix), colors.hue_helper(int(numPix), _MAX_HEIGHT, 0))
    
    # finally update the matrix
    led.update()

def audio_in():
    '''Control the lightshow from audio coming in from a USB audio card'''
    sample_rate = 48000 # 48000 Hz
    input_channels = 2 # 1 for mono input (audio in)

    # Open the input stream from default input device
    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()

    for i in range(0,p.get_device_count()):
        logging.debug("device " + str(i) + " info " + str(p.get_device_info_by_index(i)))

    logging.debug(" default input device " + str(p.get_default_input_device_info()))
    stream = p.open(format=FORMAT,channels=1,rate=sample_rate,input=True,frames_per_buffer=_CHUNK_SIZE)

    logging.debug("Running in audio-in mode - will run until Ctrl+C is pressed")
    print "Running in audio-in mode, use Ctrl+C to stop"
    try:
        frequency_limits = calculate_channel_frequency(_MIN_FREQUENCY, _MAX_FREQUENCY)

        # Start with these as our initial guesses - will calculate a rolling mean / std
        # as we get input data.
        mean = [12.0 for _ in range(_MAX_BARS)]
        std = [0.5 for _ in range(_MAX_BARS)]
        recent_samples = np.empty((250, _MAX_BARS))
        num_samples = 0

        # Listen on the audio input device until CTRL-C is pressed
        while True:
            l = 1
            data = stream.read(_CHUNK_SIZE)
 
            if l:
                try:
                    matrix = calculate_levels(data, _CHUNK_SIZE, sample_rate, frequency_limits, input_channels, _MAX_BARS)
                    if not np.isfinite(np.sum(matrix)):
                        # Bad data --- skip it
                        continue
                except ValueError as e:
                    # TODO(todd): This is most likely occuring due to extra time in calculating
                    # mean/std every 250 samples which causes more to be read than expected the
                    # next time around.  Would be good to update mean/std in separate thread to
                    # avoid this --- but for now, skip it when we run into this error is good
                    # enough ;)
                    logging.debug("skipping update: " + str(e))
                    continue

                update_lights(matrix, mean, std)

                # Keep track of the last N samples to compute a running std / mean
                #
                # TODO(todd): Look into using this algorithm to compute this on a per sample basis:
                # http://www.johndcook.com/blog/standard_deviation/
                if num_samples >= 250:
                    no_connection_ct = 0
                    for i in range(0, _MAX_BARS):
                        mean[i] = np.mean([item for item in recent_samples[:, i] if item > 0])
                        std[i] = np.std([item for item in recent_samples[:, i] if item > 0])

                        # Count how many channels are below 10, if more than 1/2, assume noise (no connection)
                        if mean[i] < 10.0:
                            no_connection_ct += 1

                    # If more than 1/2 of the channels appear to be not connected, turn all off
                    if no_connection_ct > _MAX_BARS / 2:
                        logging.debug("no input detected, turning all lights off")
                        mean = [20 for _ in range(_MAX_BARS)]
                    else:
                        logging.debug("std: " + str(std) + ", mean: " + str(mean))
                    num_samples = 0
                else:
                    for i in range(0, _MAX_BARS):
                        recent_samples[num_samples][i] = matrix[i]
                    num_samples += 1

    except KeyboardInterrupt:
        pass
    finally:
        print "\nStopping"
        stream.stop_stream()
        stream.close()
        led.all_off()
        led.update()        
        p.terminate()


if __name__ == "__main__":

    from bibliopixel.drivers.network import DriverNetwork
    #Load driver for your hardware, visualizer just for example
    #from bibliopixel.drivers.visualizer import DriverVisualizer
    from bibliopixel.drivers.network import DriverNetwork
    
    import bibliopixel.gamma as gamma
    
    w = _MAX_BARS
    h = _MAX_HEIGHT
    print "Pixel Count: {}".format(w*h)

    #driver = DriverVisualizer(width = w, height = h, stayTop = True)
    driver = DriverNetwork(num=w*h, width=w, height=h, host = _SERVER_IP)
     
    #load the LEDMatrix class
    from bibliopixel.led import *
    #change rotation and vert_flip as needed by your display
    #meine Matrix mit 12 * 20 
    led = LEDMatrix(driver, width = h, height = w, rotation = MatrixRotation.ROTATE_270, vert_flip = True)
    # Matrix fuer den Visualizer
    #led = LEDMatrix(driver, width = w, height = h, rotation = MatrixRotation.ROTATE_0, vert_flip = False)
    led.setMasterBrightness(255)
    import bibliopixel.log as log

    # Log everything to our log file
    #logging.basicConfig(filename= 'FFT_Audio.dbg',
    #                    format='[%(asctime)s] %(levelname)s {%(pathname)s:%(lineno)d}'
    #                    ' - %(message)s',
    #                    level=logging.DEBUG)

    # do audio processing
    audio_in()

 
