import librosa
import panns_inference
import pprint
from panns_inference import AudioTagging, SoundEventDetection, labels, config

chewing = 54

def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

class AudioAnalysis(object):
    def __init__(self, audio_path='resources/R9_ZSCveAHg_7s.wav'):
        self.audio_path = audio_path

    @lazy_property
    def audio(self):
        (audio, _) = librosa.core.load(self.audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        return audio

    @lazy_property
    def clipwise_output(self):
        at = AudioTagging(checkpoint_path=None, device='cuda')
        (clipwise_output, embedding) = at.inference(self.audio)
        return clipwise_output

    @lazy_property
    def framewise_output(self):
        sed = SoundEventDetection(checkpoint_path=None, device='cuda')
        framewise_output = sed.inference(self.audio)
        return framewise_output

    @lazy_property
    def detected(self):
        detected = {}
        for index, detection in enumerate(self.clipwise_output[0]):
            if detection > .5:
                label_name = config.labels[index]
                detected[label_name] = {'value': index,
                                        'detection_level': detection}
        return detected

    def __str__(self):

        return pprint.pformat(self.detected)


