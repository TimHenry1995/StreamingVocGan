import torch 
import numpy as np
import math
import warnings
from typing import Union, List

class StreamingModule():
    """Super class for streaming support for torch.nn Modules.
    
    Assumptions:
    - Each streaming module has an initial, middle and final phase and assumes that the streaming input is long enough for all three phases to occur at least once."""

    __is_final_slice__ = False
    precision = torch.float32

    @staticmethod
    def final_slice() -> None:
        """Indicates to the forward method that the next slice is the final slice.
        
        Postcondition:
        - StreamingModule.__is_final_slice__ == True.
        
        Inputs:
        - None
        
        Outputs:
        - None"""
        StreamingModule.__is_final_slice__ = True

    @staticmethod
    def close_stream() -> None:
        """Undos the operation of final_slice().
        
        Postcondition:
        - StreamingModule.__is_final_slice__ == False.
        
        Inputs:
        - None
        
        Outputs:
        - None"""
        StreamingModule.__is_final_slice__ = False

    def __init__(self):
        """Constructor for this class.
        
        Postcondition:
        self.__state__ == None.
        
        Inputs:
        - self: The object to be initialized.
        
        Outputs:
        - self: The initialized object"""
        self.__state__ = None

    def __ensure_module_precision__(self) -> None:
        """Ensures the parameters of self have the precision specified by static attribute precision.
        
        Precondition:
        - Precision of self._parameters (if exists) has some value.
        
        Postcondition:
        - Precision of self._parameters (if exists) has the value specified by the static attribute precision."""

        if hasattr(self, "_parameters"):
            for (name, parameter) in self._parameters.items():
                if not parameter.dtype == StreamingModule.precision:
                    self._parameters[name] = parameter.to(StreamingModule.precision)

    def __ensure_data_precision__(x:  Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Ensures the input has the precision specified by static attribute precision.
        
        Inputs:
        - x: input with original precision.
        
        Outputs:
        - x: x with new precision."""
        # Cast input to higher precision
        if type(x) == type([]) or type(x) == type(()):
            for i in range(len(x)):
                if not x[i].dtype == StreamingModule.precision:
                    x[i] = x[i].to(StreamingModule.precision)
        else: 
            if not x.dtype == StreamingModule.precision:
                 x = x.to(StreamingModule.precision)

        # Outputs
        return x

    def __call__(self, input: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Provides streaming support for the forward method of the regular torch.nn.Module.
        Expects that the last call to this function is preceded by statically calling final_slice() and succeded by calling close_stream().

        Precondition:
        - self.__state__ may be None, a tensor or a list of tensors.

        Postcondition:
        - self.__state__ may have changed to be None, a tensor or a list of tensors.

        Inputs:
        - input: input tensor of size [batch size, input channel count, time point count of original input slice].
        
        Outputs:
        - y_hat: output tensor of size [batch size, input channel count, combination of time point count of state and original input slice]."""
        
        # Set precision of operations
        self.__ensure_module_precision__()
        input = StreamingModule.__ensure_data_precision__(x=input)
        
        # Initial call
        if self.__state__ == None: 
            y_hat = self.__forward_and_initialize_state__(x=input)
        # Final call
        elif StreamingModule.__is_final_slice__: 
            y_hat = self.__forward_and_finish_state__(x=input)
        # Intermediate call
        else:
            y_hat = self.__forward_and_propagate_state__(x=input)

        # Output
        return y_hat

    def __forward_and_initialize_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Sets the internal state to an initial value and compute the forward operation on x.
                        
        Precondition:
        - self.__state__ is None.

        Postcondition:
        - self.__state__ is some tensor or list of tensors.

        Inputs:
        - x: input tensor of size [batch size, input channel count, time point count of original input slice].
        
        Outputs:
        - y_hat: output tensor of size [batch size, input channel count, combination of time point count of state and original input slice]."""
        pass

    def __forward_and_propagate_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Sets the state to an updated value and computes the forward operation on x.
        
        Precondition:
        - self.__state__ is some tensor or list of tensors.

        Postcondition:
        - self.__state__ is some different tensor or list of tensors.

        Inputs: 
        - x: input tensor of size [batch size, input channel count, time point count of original input slice].
        
        Outputs:
        - y_hat: output tensor of size [batch size, input channel count, combination of time point count of state and original input slice]."""
        pass

    def __forward_and_finish_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Sets the state to a final value and computes the forward operation on x. 

        Postcondition:
        - self.__state__ == None
        - All dynamic attributes are reset to start the stream again.
        
        Inputs:
        - x: input tensor of size [batch size, input channel count, time point count of original input slice].
        
        Outputs:
        - y_hat: output tensor of size [batch size, input channel count, combination of time point count of state and original input slice]."""
        pass

class StreamingConv1d(StreamingModule, torch.nn.Conv1d):
    """Provides streaming support for the Conv1d class of the torch.nn module version 1.7.1. 
    It runs a valid 1d convolution, i.e. no padding is applied to the input."""

    INSTANCE_AXIS = 0
    CHANNEL_AXIS = 1
    TIME_AXIS = 2 

    def __init__(self, **kwargs):
        """Constructor for this class.
        
        Postcondition:
        - self.original_padding == [0] or, if provided, a one element list with the value from **kwargs.
        - all other fields initialized by the constuctors of StreamingModule and Conv1d.

        Inputs:
        - self: The object to be initialized.
        - **kwargs: The same arguments as for a regular torch.nn.Conv1d.
        
        Outputs:
        - self: The instance of this class."""
        # Manage padding
        # Self should store a zero value for padding because the forward operation should not pad every time a slice is passed through
        # The field self.original_padding is used to pad with zeros in the first and last slice.
        self.original_padding = [0]
        if "padding" in kwargs.keys():
            if type(kwargs["padding"]) == int:
                self.original_padding = [kwargs["padding"]]
            else:
                self.original_padding = kwargs["padding"]
        kwargs["padding"] = [0]

        # Super construction
        torch.nn.Conv1d.__init__(self, **kwargs)
        StreamingModule.__init__(self)
        
    def __forward_slice__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Performs the forward operation simply on x. It does not manage the state and it does not add padding around x.
        It is save to call it on x which are too short to provide any output. In that case an empty tensor is returned.

        Inputs: 
        - x: The input tensor to the forward operation. 
        
        Outputs:
        - y_hat: The prediction obtained from the forward operation. 
        """
        
        # If the current input is too short, we cannot yet output any time points
        time_point_count = x.size()[self.TIME_AXIS]
        if time_point_count < self.__i_o_ratio__(): # time point count of the dilated kernel
            instance_count = x.size()[self.INSTANCE_AXIS]
            y_hat = torch.empty(instance_count, self.out_channels, 0)
        
        # Otherwise we just compute the output
        else: 
            y_hat = torch.nn.Conv1d.forward(self=self, input=x)

        # Output
        return y_hat

    def __output_to_input_index__(self, output_index: int, input_length: int = None) -> int:
        """Computes the starting time point (input_index) for the input sequence that yields the output at time point output_index.
        
        Inputs:
        - output_index: The index of the output time point for which the input time point shall be computed. If non_negative then input_length can be omitted. If negative then input_length shall be provided.
        - input_length: The number of time points of the input.
        
        Outputs:
        - input_index: The starting index for the input sequence that yields the output at time point output_index. 
            This index always counts from the start. If it is negative then the input_length is too short for an output."""
        
        # When output index is counting from start
        if 0 <= output_index:
            input_index = self.stride[0] * input_index
        
        # When output index is counting from end
        else: 
            # Compute number of time points for the output 
            # https://pytorch.org/docs/1.7.1/generated/torch.nn.Conv1d.html?highlight=conv1d#torch.nn.Conv1d
            numerator = input_length + self.padding[0]*2 - self.__i_o_ratio__() - 2
            y_time_point_count = math.floor(numerator/self.stride[0]) + 1
            
            # Adjust by output_index
            y_time_point_count += output_index
            
            # Compute corresponding input index
            input_index = self.stride[0]*(y_time_point_count) 

        # Output
        return input_index

    def __i_o_ratio__(self) -> int:
        """Computes the number of time points needed from the input sequence to get one output time point"""
        return self.dilation[0] * (self.kernel_size[0] - 1) + 1

    def __save_state__(self, x: torch.Tensor) -> None:
        """Mutator method that saves the last few time points of x into self.__state__. If x is too short to yield any output, all time points of x are kept as state.
        If called in contiguous iterations this behaviour will accumulate slices in the state until the first output can be computed.
        If the input is long enough to yield an input, then only the last few time points are saved such that the first time point of the saved state
        is the first time point needed for the last output time point. The state is never longer than what is needed to create one output time point.
        
        Postcondition:
        - self.__state__ contains the last few time points of x.

        Inputs:
        - x: The tensor from which a state shall be determined.
        
        Outputs:
        - None. """
        # The state consists of the last few time points of the input
        # It builds up over iterations until it is long enough to yield at least one output time point
        input_index = self.__output_to_input_index__(output_index=-1, input_length=x.size()[self.TIME_AXIS]) # Temporal axis
        input_index = max(0, input_index) # If the input is too short for an output we keep all x in the state
        self.__state__ = x[:,:,input_index:] # All instances, all channels, last few time points

    def __crop_input__(self, x: torch.Tensor) -> torch.Tensor:
        """Removes the last few time points from x such that it yields the first t-1 outputs where t is the number of outputs that x would normally yield.
        If x is too short to yield any output then an empty tensor will be returned.
        
        Inputs:
        - x: The input tensor that should be cropped.
        
        Outputs:
        - x: The cropped input tensor."""

        # The forward operation waits until it can complete at least one output time point
        input_index = self.__output_to_input_index__(output_index=-2, input_length=x.size()[self.TIME_AXIS]) # Temporal axis
        
        # The penultimate output time point needs these exta input time points
        input_end_index = input_index + self.__i_o_ratio__() 

        # Take all instances, all channels, first few time points
        x = x[:,:,:input_end_index] 

        # Output
        return x

    def __forward_and_initialize_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Trivial case
        if x.size()[self.TIME_AXIS] == 0: return torch.empty(x.size()[self.INSTANCE_AXIS], self.out_channels, 0)

        # Pad input
        instance_count = x.size()[self.INSTANCE_AXIS]
        zeros = torch.zeros([instance_count, self.in_channels, self.original_padding[0]])
        x = torch.cat([zeros, x], dim=self.TIME_AXIS)

        # Initialize the state
        self.__save_state__(x=x)

        # Forward
        x = self.__crop_input__(x=x)
        y_hat = self.__forward_slice__(x=x)
        
        # Output
        return y_hat

    def __forward_and_propagate_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Trivial case
        if x.size()[self.TIME_AXIS] == 0: return torch.empty(x.size()[self.INSTANCE_AXIS], self.out_channels, 0)
        
        # Concatenate input with state
        x = torch.cat([self.__state__,x], dim=self.TIME_AXIS) # Temporal axis
        
        # Update state
        self.__save_state__(x=x)

        # Forward
        x = self.__crop_input__(x=x)
        y_hat = self.__forward_slice__(x=x)
        
        # Output
        return y_hat

    def __forward_and_finish_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Trivial case
        if x.size()[self.TIME_AXIS] == 0: return torch.empty(x.size()[self.INSTANCE_AXIS], self.out_channels, 0)
        
        # Concatenate input with state 
        x = torch.cat([self.__state__, x], dim=self.TIME_AXIS) # Temporal axis
        
        # Clear the state
        self.__state__ = None

        # Pad with zeros
        instance_count = x.size()[self.INSTANCE_AXIS]
        zeros = torch.zeros([instance_count, self.in_channels, self.original_padding[0]])
        x = torch.cat([x, zeros], dim=self.TIME_AXIS)

        # Forward
        y_hat = self.__forward_slice__(x=x)

        # Output
        return y_hat

class StreamingConvTranspose1d(StreamingModule, torch.nn.ConvTranspose1d):
    """Provides streaming support for the ConvTranspose1d class of the torch.nn module version 1.7.1.
    Assumptions:
    - slice sice * stride needs to be at least as large as the padding. If this assumption is not met then the streaming output and the standard output will not match at the leading and or trailing time points."""
    
    INSTANCE_AXIS = 0
    CHANNEL_AXIS = 1
    TIME_AXIS = 2

    def __init__(self, **kwargs):
        """Constructor for this class.
        
        Inputs:
        - **kwargs: the same arguments as for a regular torch.nn.Conv1d.
        
        Outputs:
        - the instance of this class."""

        # Manage padding
        # Self should store a zero value for padding because the forward operation should not pad every time a slice is passed through
        # The field self.original_padding is used to pad with zeros in the first and last slice.
        # Input padding
        self.original_padding = [0]
        if "padding" in kwargs.keys():
            if type(kwargs["padding"]) == int:
                self.original_padding = [kwargs["padding"]]
            else:
                self.original_padding = kwargs["padding"]
        kwargs["padding"] = [0]

        torch.nn.ConvTranspose1d.__init__(self, **kwargs)
        StreamingModule.__init__(self)

    def __forward_slice__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Performs the forward operation simply on x. It does not manage the state and it does not add padding around x.
        The output will have max(0, self.stride[0] - self.__i_o_ratio__()) trailing occurences of the bias.

        Inputs: 
        - x: The input tensor to the forward operation. 
        
        Outputs:
        - y_hat: The prediction obtained from the forward operation. 
        """
        
        # Casual forward
        y_hat = torch.nn.ConvTranspose1d.forward(self=self, input=x)

        # Add trailing bias if needed
        trailing_bias_count = max(0, self.stride[0] - self.__i_o_ratio__()) # Time axis
        zeros = torch.zeros(size=[y_hat.size()[self.INSTANCE_AXIS], y_hat.size()[self.CHANNEL_AXIS], trailing_bias_count])
        bias = self.bias.unsqueeze(self.INSTANCE_AXIS).unsqueeze(self.TIME_AXIS) # Expanding bias (channel axis) along instance and time axis
        bias, _ = torch.broadcast_tensors(bias, zeros) 
        y_hat = torch.cat([y_hat, bias], dim=self.TIME_AXIS)

        # Output
        return y_hat

    def __input_to_output_index__(self, input_index: int, output_length: int = None) -> int:
        """Computes the starting time point (output_index) for the output sequence yielded from the input at time point input_index.
        
        Inputs:
        - input_index: The index of the input time point for which the output time point shall be computed. May count backwards from end (hence be negative).
        - output_length: The length of the output sequence. Only needed when the provided input_index is negative.

        Outputs:
        - output_index: The starting index for the output sequence yielded by the input at time point input_index. 
            This index always counts from the start. If it is negative then the input_index provided is negative and absolutely larger than the provided input_length."""
        
        # When input_index is counting from start
        if 0 <= input_index:
            output_index = input_index * self.stride[0] 
        
        # When input_index is counting from end
        else: 
            # Compute the input_length
            input_length = (output_length - (self.__i_o_ratio__() - self.stride[0])) // self.stride[0]

            # Compute corresponding output index for input index
            output_index = (input_length + input_index) * self.stride[0] 
            
        # Output
        return output_index

    def __i_o_ratio__(self) -> int:
        """Computes the number of time points obtained for the output sequence based on one input time point.
        
        Inputs:
        - None
        
        Outputs:
        -None"""
        return self.dilation[0] * (self.kernel_size[0] - 1) + 1

    def __save_state__(self, y_hat: torch.Tensor) -> None:
        """Mutator method that saves the time points t+1 until end of y_hat into self.__state__. 
        Here t+1 is the index in the current slice's output of the first output time point obtained from the first input time point of the next slice. 
        
        Postcondition:
        - self.__state__ contains the last few time points from y_hat.

        Inputs:
        - y_hat: The tensor from which a state shall be determined.
        
        Outputs:
        - None. """

        # 1. We only want to keep the output that overlaps with the next slice's first output time point
        # 1.1 Starting time point in output based on final input time point
        output_index = self.__input_to_output_index__(input_index=-1, output_length=y_hat.size()[self.TIME_AXIS]) # Temporal axis
        
        # 1.2 Add the stride to get the index after which the current slice's output overlaps with the next slice's output
        output_index += self.stride[0] 

        # Take all instances, all channels, first few time points
        self.__state__ = y_hat[:,:,output_index:] 
        
        # We do not want the bias to be added twice when the outputs of two consecutive slcies are summed at the intersection
        self.__subtract_bias_from_state__()

    def __crop_output__(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Crops the output such that only those first t time points remain that will not be affected by the next slice.
        Here t+1 is the index in the current slice's output of the first output time point obtained from the first input time point of the next slice. 
        
        Inputs:
        - y_hat: The output tensor that should be cropped.
        
        Outputs:
        - y_hat: The cropped output tensor."""

        # 1. We only want to keep the output that does not overlap with the next slice's output
        # 1.1 Starting time point in output based on final input time point
        output_index = self.__input_to_output_index__(input_index=-1, output_length=y_hat.size()[self.TIME_AXIS]) # Temporal axis
        
        # 1.2 Add the stride to get the index after which the current slice's output overlaps with the next slice's output
        output_index += self.stride[0] 

        # Take all instances, all channels, first few time points
        y_hat = y_hat[:,:,:output_index] 

        # Output
        return y_hat

    def __subtract_bias_from_state__(self) -> None:
        """Mutator methods that subtracts the bias from state to prevent it from being added twice when the state is added to the output of a slice.
        
        Postcondition:
        - self.__state__ is now reduced by self.bias.

        Inputs:
        - None
        
        Outputs:
        - None"""
        bias = self.bias.unsqueeze(self.INSTANCE_AXIS).unsqueeze(self.TIME_AXIS) # Expanding bias (channel axis) along instance and time axis
        bias, _ = torch.broadcast_tensors(bias, self.__state__) # All instances, all channels, last few time points
        self.__state__ = self.__state__ - bias 
        
    def __pad_output__(self, y_hat: torch.Tensor, left: bool) -> torch.Tensor:
        """Padding is implemented my the standard ConvTranspose1d such that ConvTranspose1d and Conv1d will be inverses of each other
        Here this means rather than adding zeros to both sides we need to remove the leading and trailing self.originial_padding[0] time points from y_hat.

        Inputs:
        - y_hat: The output that should be padded.
        - left: Indicates whether to pad left or right.

        Outputs:
        - y_hat: The padded output.
        """

        # Check whether output has enough timepoints to be padded (due to the transpose nature padding amount to cropping)
        if y_hat.size()[self.TIME_AXIS] < self.original_padding[0]: # Padding (here implemented as a cropping of the output) requires enough output time points
            # 1. Notify user
            # If this is the final slice a simple warning will do
            if self.__is_final_slice__:
                warnings.warn(f"StreamingConvTranspose1d assumes slice sice * stride to be at least as large as the padding. This assumption is not met for the final slice and hence the streaming output and the standard output will not match for the last few time points.")
                
                # Correct by expanding the slice with zeros
                zeros = torch.zeros([y_hat.size()[self.INSTANCE_AXIS], y_hat.size()[self.CHANNEL_AXIS], self.original_padding[0] - y_hat.size()[self.TIME_AXIS]])
                y_hat = torch.cat([y_hat, zeros], dim=self.TIME_AXIS)

            # Otherwise throw an error
            else:
                raise AssertionError(f"StreamingConvTranspose1d assumes slice sice * stride to be at least as large as the padding. This assumption is not met and hence the streaming output and the standard output will not match.")
            
        # Pad, i.e. crop output
        if left:
            y_hat = y_hat[:,:,self.original_padding[0]:]
        else:
            i = y_hat.size()[self.TIME_AXIS] - self.original_padding[0]
            y_hat = y_hat[:,:,:i]

        # Output
        return y_hat

    def __forward_and_initialize_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Trivial case
        if x.size()[self.TIME_AXIS] == 0: return torch.empty(x.size()[self.INSTANCE_AXIS], self.out_channels, 0)
        
        # Forward
        y_hat = self.__forward_slice__(x=x)

        # Initialize the state 
        self.__save_state__(y_hat=y_hat)

        # Keep only the first part of the output since it will not be affected by later time points of the next slice
        y_hat = self.__crop_output__(y_hat=y_hat)

        # Pad
        y_hat = self.__pad_output__(y_hat=y_hat, left=True)
        
        # Output
        return y_hat

    def __forward_and_propagate_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Trivial case
        if x.size()[self.TIME_AXIS] == 0: return torch.empty(x.size()[self.INSTANCE_AXIS], self.out_channels, 0)
        
        # Forward
        y_hat = self.__forward_slice__(x=x)

        # Add state
        state_time_points = self.__state__.size()[-1]
        y_hat[:,:,:state_time_points] += self.__state__

        # Update state
        self.__save_state__(y_hat=y_hat)

        # Keep only the first part of the output since it will not be affected by later time points of the next slice
        y_hat = self.__crop_output__(y_hat=y_hat)

        # Output
        return y_hat

    def __forward_and_finish_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Trivial case
        if x.size()[self.TIME_AXIS] == 0: return torch.empty(x.size()[self.INSTANCE_AXIS], self.out_channels, 0)
        
        # Forward
        y_hat = torch.nn.ConvTranspose1d.forward(self=self, input=x)
        
        # Add state
        state_time_points = self.__state__.size()[self.TIME_AXIS]
        y_hat[:,:,:state_time_points] += self.__state__

        # Reset state
        self.__state__ = None
        
        # Pad output
        y_hat = self.__pad_output__(y_hat=y_hat, left=False)

        # Output
        return y_hat

class StreamingSum(StreamingModule):
    """Provides streaming support for the sum operation for two tensors.
    Assumptions:
    - Both tensors being summed always need to be provided in the same order to the forward method.
    - Both tensors need to match along the instance and channel dimensions."""

    INSTANCE_AXIS = 0
    CHANNEL_AXIS = 1
    TIME_AXIS = 2

    def __init__(self):
        StreamingModule.__init__(self)

    def __update_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> int:
        """Appends all time points from x1 and x2 to the state and thereafter pops the first few time points from the state.
        The number of time points to be popped is equal to the minimum of time points from both state variables.
        
        Assumptions:
        - Assumes that x1 and x2 are always provided in the same order.

        Precondition:
        - self.__state__ has to be a list of two tensors. The tensors may be empty.

        Postcondition: 
        - self.__state__ is a list of two different tensors.
        
        Inputs:
        - x: A list of two tensors from which the state shall be saved.
        
        Outputs:
        - x1, x2: The tensor popped from the first and second state variables, respectively."""

        # Unpack inputs
        x1, x2 = x
        
        # Append inputs to the state variables
        state_1, state_2 = self.__state__
        state_1 = torch.cat([state_1, x1], dim=self.TIME_AXIS)
        state_2 = torch.cat([state_2, x2], dim=self.TIME_AXIS)
        
        # 1. Pop first few time points
        # 1.1 Determine minimum number of time points present in both state variables
        t1 = state_1.size()[self.TIME_AXIS]
        t2 = state_2.size()[self.TIME_AXIS]
        t = min(t1, t2)

        # 1.2 Pop those excess time points 
        x1 = state_1[:,:,:t]; x2 = state_2[:,:,:t]
        state_1 = state_1[:,:,t:]; state_2 = state_2[:,:,t:]
        
        # Save the new state variables to the state
        self.__state__ = (state_1, state_2)

        # Outputs
        return x1, x2
        
    def __forward_and_initialize_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Unpack x
        x1, x2 = x
        
        # Initialize the state
        if self.__state__ == None: self.__state__ = (torch.zeros(x1.size()[self.INSTANCE_AXIS], x1.size()[self.CHANNEL_AXIS], 0),
                                                     torch.zeros(x2.size()[self.INSTANCE_AXIS], x1.size()[self.CHANNEL_AXIS], 0))
        
        # Update the state
        x1, x2 = self.__update_state__(x=x)

        # Sum the inputs
        y = x1 + x2

        # Outputs
        return y

    def __forward_and_propagate_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Update the state
        x1, x2 = self.__update_state__(x=x)

        # Sum the inputs
        y = x1 + x2

        # Outputs
        return y

    def __forward_and_finish_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Initialize the state
        x1, x2 = self.__update_state__(x=x)

        # Reset state
        self.__state__ = None

        # Sum the inputs
        y = x1 + x2

        # Outputs
        return y

class StreamingReflectionPad1d(StreamingModule, torch.nn.ReflectionPad1d):
    """Provides streaming support for torch.nn.ReflectionPad1d for torch version 1.7.1.
    Here the state consists of the most recent self.padding time frames. If the sequence just begins
    the state will accumulate those time frames. As the state contains this many time frames for the first time
    reflection pad will be applied and output. Thereafter the state will continue to save the most recent self.padding
    time frames and the output will not be altered by the state. When the final slice is given and the state has self.padding 
    time frames then reflection pad is applied and output. 
    
    Assumptions:
    - The total number of time frames needs to be larger than self.padding."""

    INSTANCE_AXIS = 0
    CHANNEL_AXIS = 1
    TIME_AXIS = 2

    def __init__(self, **kwargs):
        # Super
        StreamingModule.__init__(self=self)
        torch.nn.ReflectionPad1d.__init__(self, **kwargs)

        # Fields
        self.is_warming_up = True


    def __propagate_x_through_state__(self, x: torch.Tensor) -> torch.Tensor:
        """Mutator method that propagates x through state and makes sure that the state is equal to the k most recent time frames,
        where k is at least 0 and at most self.padding + 1 time frames. Important! This method only covers the warm up and the ongoing stream. 
        The final slice which needs a right-reflection pad is not covered here. 
        
        Preconditions: 
        - self.__state__ is a tensor. Possibly empty.
        
        Inputs:
        - x: the input from which time frames shall be propagated through the state.
        
        Outputs:
        - y_hat: the output. If the total number of time frames of state and x is at most self.padding then y_hat is empty on the time axis.
        If that number is equal to self.padding + 1 for the first time, then state has just collected enough time points to output the left-reflection paded version of state and x. 
        If that number is equal to self.padding + 1 at a later time, then y_hat is just x.
        
        Postcondition:
        - self.__state__ is a tensor with 0 <= k <= self.padding + 1 time frames. While the state is still accumulating, i.e. in the beginning, k <= self.padding. Later k = self.padding + 1."""

        # Initialize
        y_hat = None

        # Propagate x through state
        tmp = torch.cat([self.__state__, x], dim=self.TIME_AXIS)
        
        # In the beginning the first few slices are just used for warm up
        if self.is_warming_up:
            # Either the current call to this method is still warming up the state
            if tmp.size()[self.TIME_AXIS] <= self.padding[0]:
                # Then we append x to the state
                self.__state__ = tmp
                
                # And we cannot yet return any data
                instance_count = self.__state__.size()[self.INSTANCE_AXIS]
                channel_count = self.__state__.size()[self.CHANNEL_AXIS]
                y_hat = torch.empty([instance_count, channel_count, 0]) 

            # Or it finishes the warm up
            else:
                # Then we remember that we warmed up the state
                self.is_warming_up = False
                
                # And we just save the self.padding + 1 most recent time frames
                i = tmp.size()[self.TIME_AXIS] - self.padding[0] - 1 # i is non-negative because tmp has more time frames than self.padding due to warm up
                self.__state__ = tmp[:,:,i:]

                # And we apply a reflection to output padded left-version of state and x
                tmp = torch.nn.ReflectionPad1d.forward(self, tmp)
                i = tmp.size()[self.TIME_AXIS] - self.padding[0]
                y_hat = tmp[:,:,:i] # Removing the right pad since the stream just started

        # Later 
        else:
            # We just save the self.padding + 1 most recent time frames
            i = tmp.size()[self.TIME_AXIS] - self.padding[0] - 1 # i is non-negative because tmp has more time frames than self.padding due to warm up
            self.__state__ = tmp[:,:,i:]
            
            # And output x
            y_hat = x

        # Outputs
        return y_hat

    def __forward_and_initialize_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Initialize state
        instance_count = x.size()[self.INSTANCE_AXIS]
        channel_count = x.size()[self.CHANNEL_AXIS]
        self.__state__ = torch.empty([instance_count, channel_count, 0]) 

        # Propagate x through state
        y_hat = self.__propagate_x_through_state__(x=x)

        # Outputs
        return y_hat

    def __forward_and_propagate_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Propagate x through state
        y_hat = self.__propagate_x_through_state__(x=x)

        # Outputs
        return y_hat

    def __forward_and_finish_state__(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Apply a right-reflection pad to the state and x
        tmp = torch.cat([self.__state__, x], dim=self.TIME_AXIS)
        y_hat = torch.nn.ReflectionPad1d.forward(self, tmp)

        # Crop to only return x and the rigth pad
        i = y_hat.size()[self.TIME_AXIS] - x.size()[self.TIME_AXIS] - self.padding[0] # Is positive because the class assumption asserts that in the end the state has more than self.padding time points.
        y_hat = y_hat[:,:,i:] # Remove the left pad

        # Reset fields
        self.is_warming_up = True

        # Outputs
        return y_hat
