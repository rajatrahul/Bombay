{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b2eb5071-b2a7-446d-8e85-4648c7326e70",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [13/Jun/2024 15:10:02] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [13/Jun/2024 15:10:10] \"POST /predict HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify, render_template\n",
    "import cv2\n",
    "import numpy as np\n",
    "import joblib\n",
    "from skimage.feature import hog\n",
    "from skimage.feature import local_binary_pattern\n",
    "\n",
    "\n",
    "def extract_hog(image):\n",
    "    if image is None or len(image.shape) != 2:\n",
    "        return np.array([])\n",
    "    hog_features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)\n",
    "    return hog_features\n",
    "\n",
    "\n",
    "def build_filters():\n",
    "    filters = []\n",
    "    ksize = 31\n",
    "    for theta in np.arange(0, np.pi, np.pi / 4):\n",
    "        for sigma in (1, 3):\n",
    "            for lambd in np.arange(0, np.pi, np.pi / 4):\n",
    "                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)\n",
    "                filters.append(kernel)\n",
    "    return filters\n",
    "\n",
    "\n",
    "def process(image, filters):\n",
    "    if image is None or len(image.shape) != 2:\n",
    "        return np.array([])\n",
    "    accum = np.zeros_like(image)\n",
    "    for kernel in filters:\n",
    "        fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel)\n",
    "        np.maximum(accum, fimg, accum)\n",
    "    return accum.flatten()\n",
    "\n",
    "\n",
    "filters = build_filters()\n",
    "\n",
    "\n",
    "def extract_lbp(image, P=8, R=1):\n",
    "    if image is None or len(image.shape) != 2:\n",
    "        return np.array([])\n",
    "    lbp = local_binary_pattern(image, P, R, method='uniform')\n",
    "    n_bins = int(lbp.max() + 1)\n",
    "    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))\n",
    "    return hist\n",
    "\n",
    "\n",
    "def extract_sift(image):\n",
    "    if image is None or len(image.shape) != 2 or image.dtype != np.uint8:\n",
    "        return np.array([])\n",
    "    sift = cv2.SIFT_create()\n",
    "    keypoints, descriptors = sift.detectAndCompute(image, None)\n",
    "    if descriptors is not None:\n",
    "        return descriptors.flatten()\n",
    "    else:\n",
    "        return np.array([])\n",
    "\n",
    "\n",
    "def extract_combined_features(image):\n",
    "    if image is None or len(image.shape) != 2:\n",
    "        return np.array([])\n",
    "\n",
    "    hog_features = extract_hog(image)\n",
    "    gabor_features = process(image, filters)\n",
    "    lbp_features = extract_lbp(image)\n",
    "    sift_features = extract_sift(image)\n",
    "\n",
    "    # Combine features ensuring they are of the same size\n",
    "    combined_features = np.hstack((hog_features, gabor_features, lbp_features, sift_features))\n",
    "\n",
    "    # Handling cases where some features might be empty by padding with zeros\n",
    "    expected_length = 128 * 128 * 4  # Example expected length, adjust as needed\n",
    "    if combined_features.size < expected_length:\n",
    "        combined_features = np.pad(combined_features, (0, expected_length - combined_features.size), 'constant')\n",
    "\n",
    "    return combined_features\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load('BombaymodelRF.pkl')\n",
    "\n",
    "# Load the filters for Gabor\n",
    "filters = build_filters()\n",
    "\n",
    "def preprocess_image(image):\n",
    "    # Resize, grayscale, normalize, and extract features\n",
    "    if image is None or image.size == 0:\n",
    "        return None\n",
    "    image = cv2.resize(image, (128, 128))\n",
    "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "    gray = gray / 255.0\n",
    "    features = extract_combined_features(gray)\n",
    "    return features.reshape(1,-1)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    if 'file' not in request.files:\n",
    "        return \"No file part\"\n",
    "    file = request.files['file']\n",
    "    if file.filename == '':\n",
    "        return \"No selected file\"\n",
    "    if file:\n",
    "        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)\n",
    "        features = preprocess_image(image)\n",
    "        if features is None or features.shape[0] == 0:\n",
    "            return jsonify({'error': 'Invalid image'})\n",
    "        prediction = model.predict(features)\n",
    "        return jsonify({'prediction': prediction[0]})\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21dcbccc-a295-4fa0-a57f-13eead565fec",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}