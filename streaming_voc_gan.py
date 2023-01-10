from model.generator import ModifiedGenerator
import torch, numpy as np
from utils.hparams import load_hparam_str
from model import streaming as streaming
import matplotlib.pyplot as plt
import math, time
import scipy as scipy
from typing import Tuple, List, Union
import os.path as path
from scipy.io import wavfile
from utils.stft import TacotronSTFT
from scipy import stats


class StreamingVocGan():
    """This class is the front end of a processing tool box for converting between waveform and mel spectrograms in a streaming context."""

    WAVE_SAMPLING_RATE = 22050
    WAVE_FRAMES_PER_HOP = 256
    WAVE_FRAMES_PER_WINDOW = 1024

    TIMING_CONVENTIONS = {
            "Seconds Per Spectrogram Window": WAVE_FRAMES_PER_WINDOW / WAVE_SAMPLING_RATE,
            "Seconds Per Spectrogram Hop": WAVE_FRAMES_PER_HOP / WAVE_SAMPLING_RATE,
    }

    def __init__(self, is_streaming: bool = True):
        """Constructor of this class.
        
        Inputs:
        - self: The instance.
        - is_streaming: Optional argument. Indicates whether this instance shall be used in streaming mode or in standard mode.
        
        Outputs:
        - self: The instance."""

        self.is_streaming = is_streaming
        model_path = "vctk_pretrained_model_3180.pt"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        self.hp = load_hparam_str(checkpoint['hp_str'])

        self.model = ModifiedGenerator(mel_channel=self.hp.audio.n_mel_channels, n_residual_layers=self.hp.model.n_residual_layers,
                            is_streaming=is_streaming, ratios=self.hp.model.generator_ratio, mult = self.hp.model.mult,
                            out_band = self.hp.model.out_channels)
        self.model.load_state_dict(checkpoint['model_g'])
        self.model.eval(inference=True)

    @staticmethod
    def waveform_to_mel_spectrogram(waveform: Union[torch.Tensor, np.array], original_sampling_rate: int, mel_channel_count: int = 80, min_frequency: float = 0.0, max_frequency: float = 8000.0) -> torch.Tensor:
        """Converts waveform audio to mel spectrogram.
        
        Inputs:
        - waveform: The audio waveform signal of shape [instance count, time frame count] or [time frame count].
        - original_sampling_rate: the sampling rate of the waveform.
        - mel_channel_count: Optional argument. The number of mel channels the final spectrogram should have.
        - min_frequency: Optional argument. The minimum frequency of the mel spectrogram.
        - max_frequency: Optional argument. The maximum frequency of the mel spectrogram.
        
        Outputs:
        - mel_spectrogram: The mel spectrogram. It adheres to StreamingVocGan.TIMING_CONVENTIONS and is of shape [instance count, mel channel count, time frame count] or [mel channel count, time frame count]."""

        # Input management
        if not type(waveform) == type(torch.Tensor): waveform = torch.Tensor(waveform)
        if len(waveform.shape) == 1: waveform = waveform.unsqueeze(0)
        waveform = waveform.to(torch.float32)
        for i in range(waveform.shape[0]): waveform[i] = waveform[i]/ torch.max(torch.abs(waveform[i]))

        # Converting step parameters according to current sampling rate
        hop_length= (int)(StreamingVocGan.WAVE_FRAMES_PER_HOP/StreamingVocGan.WAVE_SAMPLING_RATE*original_sampling_rate)
        win_length= (int)(StreamingVocGan.WAVE_FRAMES_PER_WINDOW/StreamingVocGan.WAVE_SAMPLING_RATE*original_sampling_rate)
        
        # Creating Mel Spectrogram
        stft = TacotronSTFT(filter_length=win_length, hop_length=hop_length, win_length=win_length,
            n_mel_channels=mel_channel_count, sampling_rate=original_sampling_rate, mel_fmin=min_frequency, mel_fmax=max_frequency)
        mel_spectrogram = stft.mel_spectrogram(y=waveform).squeeze()

        # Outputs
        return mel_spectrogram

    @staticmethod 
    def plot(mel_spectrogram: np.array, waveform_standard: torch.Tensor, waveform_streaming_slices: List[torch.Tensor] = None, slice_processing_times: List[float] = None) -> None:
        """Plots the inputs and outputs of mel_spectrogram_to_waveform. 
        
        Inputs:
        - mel_spectrogram: The spectrogram to be plotted. Shape = [mel channel count, time point count]. 
        - waveform_standard: The waveform to be plotted. Shape = [time point count]. 
        - waveform_streaming_slices: Optional argument. Provides slices of the second waveform to be plotted.
        - slice_processing_times: Optional argument. If waveform_streaming_slices is provided then a another plot is shown that plots for every slice its processing time. Times are assumed to be in seconds."""
        
        # Verify arguments
        if type(slice_processing_times) != type(None):
            if type(waveform_streaming_slices) == type(None):
                raise ValueError("If the argument slice_processing_times is provided, then waveform_streaming_slices has to be provided.")
            
        # Cast to numpy arrays
        waveform_standard = waveform_standard.detach().numpy()
        if waveform_streaming_slices != None: waveform_streaming_slices = [slice.detach().numpy() for slice in waveform_streaming_slices]

        # Count plots
        plot_count = 2 + (1 if type(waveform_streaming_slices) != type(None) else 0) + (1 if type(slice_processing_times) != type(None) else 0) 

        # Time ticks
        time_frame_count = mel_spectrogram.shape[-1]
        total_seconds = (time_frame_count-1) * StreamingVocGan.TIMING_CONVENTIONS["Seconds Per Spectrogram Hop"] + StreamingVocGan.TIMING_CONVENTIONS["Seconds Per Spectrogram Window"] 
        
        # Plots
        # Spectrogram
        plt.figure()
        plt.subplot(plot_count,1,1)
        plt.imshow(np.flipud(mel_spectrogram), aspect='auto')
        time_frame_count = mel_spectrogram.shape[-1]
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.ylabel("Mel Bin")
        plt.title("Spectrogram")
            
        # Standard output
        plt.subplot(plot_count,1,2)
        plt.plot(waveform_standard)
        plt.title("Standard Waveform")
        time_frame_count = waveform_standard.shape[-1]
        plt.xlim(0,time_frame_count)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.ylabel("Amplitude")

        # Streaming output
        if type(waveform_streaming_slices) != type(None):
            plt.subplot(plot_count,1,3)
            plt.plot(np.concatenate(waveform_streaming_slices, axis=-1))
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.title("Streaming Waveform")
            plt.xlim(0,time_frame_count)
            plt.ylabel("Amplitude")
        
            # Processing times
            if type(slice_processing_times) != type(None):
                plt.subplot(plot_count,1,4)
                slice_starts = np.cumsum([0] + [slice.shape[-1] for slice in waveform_streaming_slices[:-1]]) # Now the first number is the first slice's starting index (0), last number is last slice's starting index
                slice_stops = np.cumsum([slice.shape[-1] for slice in waveform_streaming_slices]) # Now the first number is the stop index (exclusive) of the first slice, last number is stop index of last slice
                slice_durations = np.array([slice.shape[-1]/StreamingVocGan.WAVE_SAMPLING_RATE for slice in waveform_streaming_slices])
                efficiencies = [slice_processing_times[i] / slice_durations[i] if slice_durations[i] > 0 else 0 for i in range(len(slice_durations))]
                plt.bar(x=slice_starts+(slice_stops-slice_starts)/2, height=efficiencies, width=slice_stops-slice_starts, fill=False)
                
                plt.axhline(y = 1, color = 'r', linestyle = '-')
                plt.xticks(ticks=[])
                plt.xlim(0,time_frame_count)
                plt.ylabel("Efficiency")
                plt.title("Streaming Efficiency")

                # Text box
                textstr = ',   '.join((
                    '   Efficiency = Processing time / Slice Duration',
                    r'5 percent trimmed average Slice Duration = %.2f sec' % (stats.trim_mean(slice_durations, 0.05)),
                    r'5 percent trimmed average Processing Time = %.2f sec' % (stats.trim_mean(slice_processing_times, 0.05))))

                props = dict(boxstyle='round', facecolor='white', alpha=0.5)

                # place a text box in upper left in axes coords
                plt.text(0, 0.9*plt.ylim()[1], textstr, fontsize=9,
                        verticalalignment='top', bbox=props)

        # Timeline for bottom plot
        time_frame_count = waveform_standard.shape[-1]
        tick_locations = np.arange(stop=time_frame_count, step=StreamingVocGan.WAVE_SAMPLING_RATE)
        tick_labels = np.arange(stop=total_seconds, step=1.0)
        plt.xticks(ticks=tick_locations, labels=tick_labels)
        plt.xlim(0,time_frame_count)
        plt.xlabel("Signal Time (seconds)")

        plt.show()

    def mel_spectrogram_to_waveform(self, mel_spectrogram: torch.Tensor, is_final_slice: bool = False) -> Tuple[torch.Tensor, float]:
        """Converts a mel spectrogram to a waveform. 
        
        Inputs:
        - mel_spectrogram: Mel spectrogram of shape [instance count, mel channel count, time point count] or [mel channel count, time point count]. Each spectrogram timeframe is assumed to adhere to StreamingVocGan.TIMING_CONVENTIONS.
        - is_final_slice: Indicates whether this is the final slice. If self.is_streaming == True then is_final_slice needs to be specfied.
        
        Outputs:
        - waveform: The time domain signal obtained from x. If the mel_spectrogram had multiple instance, waveform will have shape [instance count, time frame count], else it will have shape [time frame count].
        - processing_time: The time it took to convert the mel_spectrogram to the waveform in seconds."""
        
        # Some constants
        mel_channel_count = mel_spectrogram.shape[-2]
        tick = time.time()

        # Predict
        with torch.no_grad():
            # Unsqueeze along instance dimension if we have a single instance
            if len(mel_spectrogram.shape) == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0) # Now shape == [instance count, channel count, time frame count]
            
            # Predict
            if is_final_slice: streaming.StreamingModule.final_slice()
            
            # pad input mel to cut artifact, see https://github.com/seungwonpark/melgan/issues/8
            if (not self.is_streaming) or is_final_slice: 
                padding = torch.full((1, mel_channel_count, 10), -11.5129)
                mel_spectrogram = torch.cat((mel_spectrogram, padding), dim=2) 

            waveform = self.model.forward(mel_spectrogram)
            
            if is_final_slice: streaming.StreamingModule.close_stream()

            # Post process
            # Related to the artifact mentioned above
            if (not self.is_streaming) or is_final_slice: 
                waveform = waveform[:,:,:-(StreamingVocGan.WAVE_FRAMES_PER_HOP*10)]
            waveform = waveform.squeeze()

            processing_time = time.time() - tick # In seconds

            return waveform, processing_time

    @staticmethod
    def slice_duration_to_frame_count(spectrogram_time_frame_count, target_seconds_per_slice: float) -> Tuple[float, float, int]:
        """Helps to configure the slice generation by providing time frame count and duration for slices of a spectrogram. 
        Assumes that the spectrogram adheres to StreamingVocGan.TIMING_CONVENTIONS.
        
        Inputs:
        - spectrogram_time_frame_count: The total number of time frames in the spectrogram.
        - target_seconds_per_slice: The desired number of seconds per slice. Note that due to rastorization the actual duration might differ slightly (see output).
        
        Outputs:
        - time_frame_count_per_slice: The number of spectrogram time frames that should be used per slice.
        - actual_seconds_per_slice: The actual duration spanned by the slice in seconds.
        - slice_count: The number of slices that can be obtained with the time_frame_count. Note that all slices are assumed to have the here computed time frame count, expect for the last slice which may be shorter."""

        # Compute number of time frames
        a =  StreamingVocGan.TIMING_CONVENTIONS["Seconds Per Spectrogram Window"] 
        b = StreamingVocGan.TIMING_CONVENTIONS["Seconds Per Spectrogram Hop"] 
        time_frames_per_slice = (int)(np.round((target_seconds_per_slice - a)/b)) + 1
        
        # Actual duration of slice
        actual_seconds_per_slice = (time_frames_per_slice-1) * b + a

        # Number of slices
        slice_count = math.ceil(spectrogram_time_frame_count/time_frames_per_slice)

        # Outputs
        return time_frames_per_slice, actual_seconds_per_slice, slice_count

    @staticmethod
    def slice_generator(mel_spectrogram: torch.Tensor, time_frames_per_slice: float) -> Tuple[torch.Tensor, bool]:
        """Generates slices of the input tensor x such that they can be fed to the convert function.
        
        Inputs:
        - mel_spectrogram: The input mel spectrogram to convert with all time frames. Shape == [mel channel count, time frame count]
        - time_frame_count_per_slice: The slice size in time frames. 
        
        Outputs:
        - x_i: A slice of the input x. Time frame count is always equal to slice_size except for the last slice whose lenght k is 0 < k <= slice_size.
        - is_final_slice: Indicates whether this slice is the final one."""

        # Step management
        time_frame_count = mel_spectrogram.shape[-1]
        slice_count = math.ceil(time_frame_count/ time_frames_per_slice)

        # Iterator
        for i in range(slice_count):
            x_i = mel_spectrogram[:,i*time_frames_per_slice:(i+1)*time_frames_per_slice]
            is_final_slice = i == slice_count - 1
            # Outputs
            yield (x_i, is_final_slice)

    @staticmethod
    def save(waveform: torch.Tensor, file_path: str) -> None:
        """Saves the audio to file. 
        
        Calling Instruction:
        - Call this method only on the entire sequence, rather than individual snippets. This method scales the audio such that its maximum absolute value is equal to the maximum absolute value of int16.
        
        Inputs:
        - audio: The audio sequence to be saved. Shape = [time frame count]. Assumed to be sampled at self.SAMPLING_RATE.
        - file_path: The path to the target file including file name and .wav extension.

        Outputs:
        - None
        """
        # Cast to numpy
        waveform = waveform.detach().numpy()

        # Scale
        amplitude = np.iinfo(np.int16).max
        waveform = (amplitude*(waveform/np.max(np.abs(waveform)))).astype(np.int16)
        
        # Save
        scipy.io.wavfile.write(filename=file_path, rate=StreamingVocGan.WAVE_SAMPLING_RATE, data=waveform)
    
if __name__ == "__main__":
    # File configuration
    file_name = 'Dutch example.wav'
    input_path = path.join('original_audio_files/',file_name)
    output_path_standard = path.join('generated_audio_files/standard',file_name)
    output_path_streaming = path.join('generated_audio_files/streaming',file_name)

    # Load waveform
    samplerate, waveform = wavfile.read(input_path)
    print(f"The audio is {waveform.shape[-1]/samplerate} seconds long.")

    # Convert to spectrogram
    # It is also possible to use your own mel spectrogram. Be sure it adheres to StreamingVocGan.TIMING_CONVENTIONS
    mel_spectrogram = StreamingVocGan.waveform_to_mel_spectrogram(waveform=waveform, original_sampling_rate=samplerate)
    
    # Standard Demo (stadard means no streaming)
    # Load model
    standard_model = StreamingVocGan(is_streaming=False)

    # Convert
    waveform_standard, standard_processing_time = standard_model.mel_spectrogram_to_waveform(mel_spectrogram=mel_spectrogram)
    print(f"Converting the spectrogram to waveform took {standard_processing_time} seconds in standard mode.")

    # Save
    StreamingVocGan.save(waveform=waveform_standard, file_path=output_path_standard)

    # Streaming Demo
    # Load model
    streaming_model = StreamingVocGan(is_streaming=True)

    # Setup a generator for the spectrogram slices
    time_frames_per_slice, actual_seconds_per_slice, slice_count = StreamingVocGan.slice_duration_to_frame_count(spectrogram_time_frame_count=mel_spectrogram.shape[-1], target_seconds_per_slice=0.075)
    print(time_frames_per_slice)
    print(f"The duration of each spectrogram slice is {actual_seconds_per_slice} seconds.") # Note, the output slices will have duration of frames_per_slice * StreamingVocGan.TIMING_CONVENTIONS['Seconds Per Spectrogram Hop'] which is a bit shorter. The surplus is saved in the state of the vocgan during streaming
    generator = streaming_model.slice_generator(mel_spectrogram=mel_spectrogram, time_frames_per_slice=time_frames_per_slice)
    
    # Containers
    waveform_streaming_slices = [None] * slice_count
    slice_processing_times = [None] * slice_count
    
    # Stream
    for i in range(slice_count):
        x_i, is_final_slice = next(generator)
        waveform_streaming_slices[i], slice_processing_times[i] = streaming_model.mel_spectrogram_to_waveform(mel_spectrogram=x_i, is_final_slice=is_final_slice)
    print(f"Converting the spectrogram to waveform took {np.sum(slice_processing_times)} seconds in streaming mode.")
    
    # Save
    waveform_streaming = torch.cat(waveform_streaming_slices, axis=-1)
    StreamingVocGan.save(waveform=waveform_streaming, file_path=output_path_streaming)

    # Plot
    StreamingVocGan.plot(mel_spectrogram=mel_spectrogram, waveform_standard=waveform_standard, waveform_streaming_slices=waveform_streaming_slices, slice_processing_times=slice_processing_times)
