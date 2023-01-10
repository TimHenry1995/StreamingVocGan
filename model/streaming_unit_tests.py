import streaming as st
import torch, math
import numpy as np
import copy as cp
from typing import Callable
import matplotlib.pyplot as plt

# Indicates whether tests should plot their results
plot = False

class TensorTestUtils:

    def evaluate_prediction(streaming_model: st.StreamingModule, standard_model: torch.nn.Module, x: torch.Tensor, slice_size: int) -> bool:
        """Predicts the output for the streaming model and its standard equivalent.
        
        Inputs:
        - streaming_model: The model used to covnert x to y during streaming
        - standard_forward: The standard model to convert x to y (no streaming).
        - x: The input data.
        - slice_size: The number of time points per slice during streaming
        
        Outputs:
        - is_equal: Indicated whether the outputs of the streaming and standard models are equal."""
        
        # Predict standard
        standard_output = standard_model(input=x)
        
        # Predict stream
        temporal_dim = 2
        time_point_count = x.size()[temporal_dim]
        streaming_outputs = []
        i = 0
        while i < time_point_count:
            x_i = x[:,:,i:i+slice_size]
            
            is_end_of_stream = i + slice_size >= time_point_count
            if is_end_of_stream:
                st.StreamingModule.final_slice()
            streaming_outputs.append(streaming_model(input=x_i))
            if is_end_of_stream:
                st.StreamingModule.close_stream()

            i += slice_size

        streaming_output = torch.cat(streaming_outputs, dim=2)
        
        # Evaluate
        is_equal = np.allclose(a=standard_output.detach().numpy(), b=streaming_output.detach().numpy(), atol=1e-06)
        
        if plot:
            plt.figure(); 
            plt.subplot(3,1,1)
            plt.plot(standard_output[0,0,:].detach().numpy()); plt.title("Standard output"); plt.ylabel("Amplitude"); plt.xticks([])
            
            plt.subplot(3,1,2)
            plt.plot(streaming_output[0,0,:].detach().numpy()); plt.title("Streaming output"); plt.ylabel("Amplitude"); plt.xticks([])
            
            plt.subplot(3,1,3)
            plt.plot((standard_output[0,0,:] - streaming_output[0,0,:]).detach().numpy()); plt.title("Standard - Streaming Output"); plt.ylabel("Amplitude"); plt.xlabel("Time")
            tmp = np.cumsum([max(0,y_i.shape[-1]) for y_i in streaming_outputs]) -1
            tmp[tmp < 0] = 0
            plt.scatter([0] + list(tmp),[0]*(1+len(streaming_outputs)), c='r'); plt.legend(["Cuts"])
            
            plt.show()

        # Output
        return is_equal

class StreamingConv1d():
    """Defines unit tests for the Streaming_Conv1d class. Tests sample from combinations of stride, dilation, padding, kernel size and slice size."""

    def forward_A():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 241
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 17
        kernel_size = 11
        dilation = 1
        stride = 5
        padding = 0
        slice_size = 3
        streaming_model = st.StreamingConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=0)
        
        # Create standard model
        standard_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=0)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test A for StreamingConv1d")

    def forward_B():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 111
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 17
        kernel_size = 14
        dilation = 3
        stride = 7
        padding = 3
        slice_size = 27
        streaming_model = st.StreamingConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test B for StreamingConv1d")

    def forward_C():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 41
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 17
        kernel_size = 1
        dilation = 3
        stride = 5
        padding = 0
        slice_size = 13
        streaming_model = st.StreamingConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test C for StreamingConv1d")

    def forward_D():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 41
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 17
        kernel_size = 1
        dilation = 3
        stride = 5
        padding = 11
        slice_size = 11
        streaming_model = st.StreamingConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test D for StreamingConv1d")

    def forward_E():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 48
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 17
        kernel_size = 5
        dilation = 4
        stride = 2
        padding = 11
        slice_size = 3
        streaming_model = st.StreamingConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test E for StreamingConv1d")

    def forward_F():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 48
        instance_count = 16
        in_channels = 32
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 17
        kernel_size = 7
        dilation = 1
        stride = 1
        padding = 0
        slice_size = 3
        streaming_model = st.StreamingConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test F for StreamingConv1d")

class StreamingConvTranspose1d():
    """Defines unit tests for the Streaming_Conv1d class"""

    def forward_A():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 16
        in_channels = 30
        torch.manual_seed(4)
        x = 0*torch.randn(instance_count, in_channels, time_point_count)+1

        # Create streaming model
        out_channels = 21
        kernel_size = 4
        stride = 3
        padding = 2
        dilation = 1
        slice_size = 3
        streaming_model = st.StreamingConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test A for StreamingConvTranspose1d")

    def forward_B():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 12
        kernel_size = 7
        stride = 2
        padding = 0
        dilation = 2
        slice_size = 1
        streaming_model = st.StreamingConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation = dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test B for StreamingConvTranspose1d")

    def forward_C():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 7
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 12
        kernel_size = 6
        stride = 2
        padding = 3
        dilation = 1
        slice_size = 1
        streaming_model = st.StreamingConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation = dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate (expect an error because the slice size is too small)
        try:
            is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
            print("Failed unit test C for StreamingConvTranspose1d")
        except AssertionError as e:
            print("Passed unit test C for StreamingConvTranspose1d")
        except:
            print("Failed unit test C for StreamingConvTranspose1d")

    def forward_D():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 12
        kernel_size = 7
        stride = 2
        padding = 1
        dilation = 3
        slice_size = 5
        streaming_model = st.StreamingConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test D for StreamingConvTranspose1d")
    
    def forward_E():
        """Tests the forward method of this class."""

        # Create example data
        time_point_count = 123
        instance_count = 19
        in_channels = 26
        x = torch.randn(instance_count, in_channels, time_point_count)

        # Create streaming model
        out_channels = 12
        kernel_size = 7
        stride = 2
        padding = 10
        dilation = 2
        slice_size = 5
        streaming_model = st.StreamingConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation = dilation, padding=padding)
        
        # Create standard model
        standard_model = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        standard_model.weight = streaming_model.weight
        standard_model.bias = streaming_model.bias

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test E for StreamingConvTranspose1d")

class StreamingSum():
    def forward_A():
        # Define some variables
        slice_sizes_1 = [0,5,12,13,19,23,30] # Needs to accumulate to same sum as slice_sizes_2
        slice_sizes_2 = [0,2,4,11,15,27,30]
        instance_count = 16
        channel_count = 32
        time_point_count = slice_sizes_1[-1]

        # Create dummy data
        x1 = torch.rand(size=[instance_count, channel_count, time_point_count])
        x2 = torch.rand(size=[instance_count, channel_count, time_point_count])

        # Stream
        streaming_model = st.StreamingSum()
        standard_output = x1 + x2
        streaming_outputs = []
        
        for i in range(len(slice_sizes_2)-1):

            x1_i = x1[:,:,slice_sizes_1[i]:slice_sizes_1[i+1]]
            x2_i = x2[:,:,slice_sizes_2[i]:slice_sizes_2[i+1]]
            is_end_of_stream = i == len(slice_sizes_1) - 2
            if is_end_of_stream: st.StreamingModule.final_slice()
            streaming_outputs.append(streaming_model(input=[x1_i, x2_i]))
            if is_end_of_stream: st.StreamingModule.close_stream()

        streaming_output = torch.cat(streaming_outputs, dim=2)

        # Evaluate
        is_equal = np.allclose(a=standard_output.detach().numpy(), b=streaming_output.detach().numpy(), atol=1e-06)
        print("Passed" if is_equal else "Failed", "unit test A for StreamingSum")

class StreamingReflectionPad1d():
    
    def forward_A():
        # Generate data
        time_point_count = 94
        x = torch.rand([13,15,time_point_count])
        padding = 3
        slice_size = 11
        
        # Generate models
        standard_model = torch.nn.ReflectionPad1d(padding=padding)
        streaming_model = st.StreamingReflectionPad1d(padding=padding)

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test A for StreamingReflectionPad1d")

    def forward_B():
        # Generate data
        time_point_count = 123
        x = torch.rand([17,11,time_point_count])
        padding = 30
        slice_size = 11
        
        # Generate models
        standard_model = torch.nn.ReflectionPad1d(padding=padding)
        streaming_model = st.StreamingReflectionPad1d(padding=padding)

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test B for StreamingReflectionPad1d")


    def forward_C():
        # Generate data
        time_point_count = 23
        x = torch.rand([17,11,time_point_count])
        padding = 20
        slice_size = 1
        
        # Generate models
        standard_model = torch.nn.ReflectionPad1d(padding=padding)
        streaming_model = st.StreamingReflectionPad1d(padding=padding)

        # Evaluate
        is_equal = TensorTestUtils.evaluate_prediction(streaming_model=streaming_model, standard_model=standard_model, x=x, slice_size=slice_size)
        print("Passed" if is_equal else "Failed", "unit test C for StreamingReflectionPad1d")

if __name__ == "__main__":
    
    # Conv1d
    StreamingConv1d.forward_A()
    StreamingConv1d.forward_B()
    StreamingConv1d.forward_C()
    StreamingConv1d.forward_D()
    StreamingConv1d.forward_E()
    StreamingConv1d.forward_F()
    
    # ConvTranspose1d
    StreamingConvTranspose1d.forward_A()
    StreamingConvTranspose1d.forward_B()
    StreamingConvTranspose1d.forward_C()
    StreamingConvTranspose1d.forward_D()
    StreamingConvTranspose1d.forward_E()

    # Sum
    StreamingSum.forward_A()
    
    # ReflectionPad1d
    StreamingReflectionPad1d.forward_A()
    StreamingReflectionPad1d.forward_B()
    StreamingReflectionPad1d.forward_C()
    