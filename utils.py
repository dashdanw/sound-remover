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

    @property
    def audio(self):
        if not hasattr(self, '_audio'):
            (audio, _) = librosa.core.load(self.audio_path, sr=32000, mono=True)
            self._audio = audio[None, :]  # (batch_size, segment_samples)
        return self._audio

    @property
    def clipwise_output(self):
        if not hasattr(self, '_clipwise_output'):
            at = AudioTagging(checkpoint_path=None, device='cuda')
            (clipwise_output, embedding) = at.inference(self.audio)
            self._clipwise_output = clipwise_output
        return self._clipwise_output

    @property
    def framewise_output(self):
        if not hasattr(self, '_framewise_output'):
            sed = SoundEventDetection(checkpoint_path=None, device='cuda')
            framewise_output = sed.inference(self.audio)
            self._framewise_output = framewise_output
        return self._framewise_output

    def __str__(self):
        detected = {}
        for index, detection in enumerate(self.clipwise_output[0]):
            if detection > .5:
                label_name = config.labels[index]
                detected[label_name] = {'value': index,
                                        'detection_level': detection}
        return pprint.pformat(detected)


