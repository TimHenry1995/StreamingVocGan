# StreamingVocGan

# Usage
Install the required packages using for instance pip install -r requirements.txt. Then download the pertained model via https://drive.google.com/file/d/1nfD84ot7o3u2tFR7YkSp2vQWVnNJ-md_/view?usp=sharing described here https://datashare.ed.ac.uk/handle/10283/3443 . Lastly, see the main script of streaming_voc_gan.py for a demonstration of the streaming vocgan.
If you are interested in using our custom streaming modules within your pytorch projects you can use them from the streaming.py module. This code is documented and unit tested.

# Credits
This code is based on the non-streaming implementation of VocGan found https://github.com/rishikksh20/VocGAN and documented https://arxiv.org/abs/2007.15256 . The added benefit of this repository is the support for streaming. The majority of the new code is in model/streaming.py 
