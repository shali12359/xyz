import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
NUMBER_SAMPLES_TO_CONSIDER = 44100

class _keyword_spotting_service:

  model = None
  _mappings = [
     "zero",
      "three",
      "nine",
      "twohundrend_twenty_two",
      "thirty",
      "fifty_four",
      "eighty_eight",
      "sixhundrend_thirty_four",
      "seven",
      "fifteen"
  ]
  _instance = None

  def predict(self, file_path):
    # extract MFCC
    MFCCs = self.preprocess(file_path)

    # convert 2d MFCCs array into 4d array
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

    # make prediction
    predictions = self.model.predict(MFCCs)
    predicted_index = np.argmax(predictions)
    predicted_keyword = self._mappings[predicted_index-1]

    return predicted_keyword

  def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    # load audio files
    signal, sr = librosa.load(file_path)

    # ensure consistency of audio
    if len(signal) > NUMBER_SAMPLES_TO_CONSIDER:
      signal =  signal[:NUMBER_SAMPLES_TO_CONSIDER]

    # extract mfcc
    MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    return MFCCs.T

def keyword_spotting_service():
  # ensure only have one instance of KSS
  if _keyword_spotting_service._instance is None:
    _keyword_spotting_service._instance = _keyword_spotting_service()
    _keyword_spotting_service.model = keras.models.load_model(MODEL_PATH)

  return _keyword_spotting_service._instance

if __name__ == "__main__":
  kss = keyword_spotting_service()

  keyword = kss.predict("7_55.wav")

  print(f"Predicted keyword: {keyword}")