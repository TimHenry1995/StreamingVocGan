from model.generator import ModifiedGenerator
import torch
from utils.hparams import load_hparam_str
import matplotlib.pyplot as plt
import numpy as np
import wave, time
import model.streaming as st
import pyaudio, math
from scipy.io.wavfile import write

# Configuration
plot = True

def play_audio(file_name):
    CHUNK = 1024

    with wave.open(file_name, 'rb') as wf:
        # Instantiate PyAudio and initialize PortAudio system resources (1)
        p = pyaudio.PyAudio()

        # Open stream (2)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Play samples from the wave file (3)
        while len(data := wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
            stream.write(data)

        # Close stream (4)
        stream.close()

        # Release PortAudio system resources (5)
        p.terminate()

def set_standard_generator_precision(generator, precision=torch.float64):
    for module in generator._modules.values():
        for (name, parameter) in module._parameters.items():
            module._parameters[name] = parameter.to(precision)
        for submodule in module._modules.values():
            for (name, parameter) in submodule._parameters.items():
                submodule._parameters[name] = parameter.to(precision)
            for subsubmodule in submodule._modules.values():
                for (name, parameter) in subsubmodule._parameters.items():
                    subsubmodule._parameters[name] = parameter.to(precision)
                for subsubsubmodule in subsubmodule._modules.values():
                    for (name, parameter) in subsubsubmodule._parameters.items():
                        subsubsubmodule._parameters[name] = parameter.to(precision)
        
def test_A():
    # Configuration
    precision = torch.float32

    # Create standard model
    model_path = "vctk_pretrained_model_3180.pt"
    device = "cpu"
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    hp = load_hparam_str(checkpoint['hp_str'])

    standard_model = ModifiedGenerator(mel_channel=hp.audio.n_mel_channels, n_residual_layers=hp.model.n_residual_layers,
                        is_streaming=False, ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels)
    standard_model.load_state_dict(checkpoint['model_g'])
    standard_model.eval(inference=True)
    set_standard_generator_precision(generator=standard_model, precision=precision)

    # Create streaming model
    streaming_model = ModifiedGenerator(mel_channel=hp.audio.n_mel_channels, n_residual_layers=hp.model.n_residual_layers,
                        is_streaming=True, ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels)
    streaming_model.load_state_dict(checkpoint['model_g'])
    streaming_model.eval(inference=True)
    st.StreamingModule.precision=precision

    # Load data
    input_path = "/Users/timdick/Documents/Master_Internship/vocgan/mel_spectrograms/LJ001-0072.wav.pt"
    x = torch.load(input_path) # shape == [spectral bins, time points]
    x = x.to(precision)
    
    # Predict
    y_hat_standard = standard_model.forward(x.unsqueeze(0)).detach().numpy()[0,0,:]

    time_frame_count = x.shape[-1]
    hop_length_seconds = 256/22050
    step_size_seconds = 4.1 * hop_length_seconds # in seconds. separates spectrogram into snippets along temporal axis
    step_size_frames = (int)(step_size_seconds/hop_length_seconds)
    step_count = math.ceil(time_frame_count/step_size_frames)
    streaming_outputs = [None] * step_count
    tick = time.time()
    
    i = 0; j = 0
    while i < time_frame_count:
        x_j = x[:,i:i+step_size_frames]

        is_last_call = j == step_count - 1
        if is_last_call:
            st.StreamingModule.final_slice()

        streaming_outputs[j] = streaming_model.forward(x_j.unsqueeze(0)).detach().numpy()
        
        if is_last_call:
            st.StreamingModule.close_stream()
        j += 1
        i += step_size_frames

    y_hat_streaming = np.concatenate(streaming_outputs, axis=-1)[0,0,:]
    tock = time.time()
    print("Required ", tock-tick, "seconds to stream convert ", time_frame_count*hop_length_seconds, "seconds of audio")
    # Evaluate
    is_equal = np.allclose(y_hat_standard, y_hat_streaming, atol=1e-06)
    print("Passed" if is_equal else "Failed", "unit test A for generator")
    
    # Plotting
    if plot:
        plt.figure(); 
        plt.subplot(3,1,1)
        plt.plot(y_hat_standard); plt.title("Standard output"); plt.ylabel("Amplitude"); plt.xticks([])
        
        plt.subplot(3,1,2)
        plt.plot(y_hat_streaming); plt.title("Streaming output"); plt.ylabel("Amplitude"); plt.xticks([])
        
        plt.subplot(3,1,3)
        tmp = np.cumsum([y_i.shape[-1] for y_i in streaming_outputs]) -1
        tmp[tmp < 0] = 0
        plt.scatter([0] + list(tmp),[0]*(1+len(streaming_outputs)), c='r')
        plt.plot(y_hat_standard - y_hat_streaming); plt.title("Standard - Streaming Output"); plt.ylabel("Amplitude"); plt.xlabel("Time")
        plt.legend(["Cuts"])
        plt.show()
    
    # Saving
    amplitude = np.iinfo(np.int16).max
    write(filename="streaming output.wav", rate=22050, data=(amplitude*(y_hat_streaming/np.max(y_hat_streaming))).astype(np.int16))
    write(filename="standard output.wav", rate=22050, data=(amplitude*(y_hat_standard/np.max(y_hat_standard))).astype(np.int16))
    write(filename="difference output.wav", rate=22050, data=(amplitude*((y_hat_standard - y_hat_streaming)/np.max(y_hat_standard - y_hat_streaming))).astype(np.int16))
    
    #play_audio(file_name="streaming_output.wav")

    k=8

if __name__ == "__main__":
    test_A()