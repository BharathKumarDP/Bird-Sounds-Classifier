## Bird-Sounds-Classifier
A Deep learning model to identify and classify Bird Sounds using CNNs.
Have you ever heard a bird’s call and wondered which bird is that? You can use our model to find it!

# Background
Bird Classification/Identification is an important ecological and social problem and the process involves
difficulties like lack of proper data, similar sounding birds,lack of proper instruments and so on. Manual
inspection is highly complex and cannot give accurate results.                                                 (Fun fact:The Great Salim Ali was the first 
person to create a proper account of Bird Life data through a nationwide survey and he did this all by himself manually!)
<p>Still,the problem is highly relevant and Ecologists need account of the population of Endangered bird species and other useful data. Ornithologists and Bird Watchers need reliable data for proper study of birds.Visual inspection is mostly impractical and difficult and so they rely on Bird Calls and Sounds.

# Abstract
The Bird Sounds are preprocessed by removing noise and converted into [Mel Spectrograms](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). These are given as input to a CNN model which is excellent in classification of images.We used Data Augmentation to increase the training data and also to make the model more robust.  


# References:
<p>  [Check out our Video Demonstration](https://www.youtube.com/watch?v=Gy_WHTzGzGg&list=PL0WyH9wLhzeknIPZwWAgktZR0oOKjCvvP&index=2)
<p>Papers</p>
<p>1.Audio Based Bird Species Identification using
Deep Learning Techniques
Elias Sprengel, Martin Jaggi, Yannic Kilcher, and Thomas Hofmann</p>
</p>2. Thanks to John Martinson for his Paper and Github repo https://github.com/johnmartinsson/bird-species-classification</p>
