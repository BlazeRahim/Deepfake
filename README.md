# Deepfake video detector (VeriFace)

A bachelor's research prototype that classifies a video as real or deepfake from the faces in it. Published as *VeriFace: Deepfake Detector Using Deep Learning*, Springer CCIS vol. 2234 (ICICBDA 2024). Follow-up work (VeriLens) is under review.

## How it works
1. **Face extraction:** sample one frame per second with OpenCV and crop faces with a Haar cascade (224×224).
2. **Features:** run each face through VGG16 (ImageNet weights, global average pooling) to get a 512-dimensional vector.
3. **Classifier:** a small Conv1D + LSTM network (`Models/lrcn.py`) scores each face; the video score is the mean over its faces.
4. **API:** a Flask endpoint (`Models/app.py`, `POST /detect`) takes an uploaded video and returns a label for the web dashboard.

## Run it
```bash
cd Models
pip install -r requirements.txt
python app.py          # serves POST /detect on http://localhost:5000
```
Training: `main.py` extracts and saves face features from `Videos/Real` and `Videos/Fake`; `lrcn.py` trains the classifier and saves `lrcn.h5`. The scripts use Windows-style paths.

## What I would do differently now
Looking back at this code with two more years of experience:
- **Split by video, not by face.** `lrcn.py` splits individual face crops at random, so faces from the same video can end up in both train and test, which makes test scores look better than they are.
- **Model time properly.** The LSTM runs over the 512 feature values of a single face. A real temporal model would take the sequence of faces across frames.
- **Report more than accuracy.** Deepfake datasets are imbalanced, so balanced accuracy, per-class recall and AUC at the video level are the honest numbers.
- **Return real scores.** The API's label comes from the model, but the percentages it sends to the dashboard are fixed placeholder values.
- **Use a modern face detector** instead of the Haar cascade.
