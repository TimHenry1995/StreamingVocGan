import os
import torch
from scipy.io.wavfile import write
import numpy as np
from model.generator import ModifiedGenerator
from utils.hparams import load_hparam_str
from denoiser import Denoiser
import pyaudio
from timeflux.core.node import Node
import time

class SpectrogramGenerator(Node):
    def __init__(self, input_path):
        self.x = torch.load(input_path) # shape == [spectral bins, time points]
        print("spec ", self.x.size())
        self.time_frame_count = self.x.shape[-1]
        self._MS_PER_SPECTRORGAM_TIME_FRAME = 256/22050
        self.step_size_ms = 10 # in milliseconds. separates spectrogram into snippets along temporal axis
        self.step_size_frames = self.step_size_ms * 1
        self.window_size = 1000 # in milliseconds
        self.i = 0

    def update(self):
                
        if self.i < self.step_count:
            x_i = self.x[:,self.i*self.stride:(self.i+1)*self.stride] # Take a temporal slice
            self.o.set(x_i.detach().numpy())
            self.i += 1
            

class Vocgan(Node):
    def __init__(self, model_path, device='cpu'):

        # Load model
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.hp = load_hparam_str(checkpoint['hp_str'])

        self.model = ModifiedGenerator(mel_channel=self.hp.audio.n_mel_channels, n_residual_layers=self.hp.model.n_residual_layers,
                            is_streaming=True, ratios=self.hp.model.generator_ratio, mult = self.hp.model.mult,
                            out_band = self.hp.model.out_channels)
        self.model.load_state_dict(checkpoint['model_g'])
        self.model.eval(inference=True)

    def update(self):
        """Expects input to be a mel spectrogram where one time frame will be translated into 256 timeframes of audio. An audio sampling rate of 22050 is assumed which means one spectrogram timeframe is about 11.6ms long"""
        
        # Make sure we have a non-empty dataframe
        if self.i.ready():
            
            # Infer the next sound snippet
            x = self.i.data # Mel spectogram
            x = torch.from_numpy(x.to_numpy())
            y_hat = self.infer(x=x) # Audio waveform
            print(" Audio: ", y_hat.shape[0])
            # Copy the input to the output
            self.o.set(y_hat)
            
    def infer(self, x, denoise=False, max_wav_value = 32768.0):
        with torch.no_grad():
            
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            
            y = self.model.inference(x)

            y = y.squeeze(0)  # collapse all dimension except time axis
            if denoise:
                denoiser = Denoiser(self.model).cuda()
                y = denoiser(y, 0.01)

            y = y.squeeze()
            y = y[:-(self.hp.audio.hop_length*10)]
            y = max_wav_value * y
            y = y.clamp(min=-max_wav_value, max=max_wav_value-1)
            y = y.short()
            y = y.cpu().detach().numpy()

            y = y/np.max(np.abs(y))
            return y.astype(np.float32)

class AudioPlayer(Node):
    data = np.array([], dtype=np.float32)
    def __init__(self):
        # instantiate PyAudio (1)
        self.p = pyaudio.PyAudio()
        self.stream = None
        
    # define callback
    def callback(in_data, frame_count, time_info, status):
        data = AudioPlayer.data[:frame_count]
        AudioPlayer.data = AudioPlayer.data[frame_count:]
        if len(data) < frame_count:
            data = np.concatenate([data, np.array([0] * (frame_count - len(data)), dtype=np.float32)], axis=0)
        return (bytes(data), pyaudio.paContinue)

    def update(self):
        if self.i.ready():
            data = np.squeeze(self.i.data.to_numpy())
            AudioPlayer.data = np.concatenate([AudioPlayer.data, data], axis=0)

            # Ensure stream is active
            #if self.stream == None:
            #    # TODO: Find out what format means
            #    self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=22050, frames_per_buffer=1024, output=True, output_device_index=1, stream_callback=AudioPlayer.callback)
            #    self.stream.start_stream()
        else:
            from scipy.io.wavfile import write
            write("example.wav", 22050, (32760*AudioPlayer.data).astype(np.int16))
        

    def terminate(self):
        super().terminate()
        self.logger.info("Terminate Audio playback Stream")
        if self.stream != None:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

