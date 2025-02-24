# Project Title: NER and Image Classification Models

## Overview
This project implements Named Entity Recognition (NER) and image classification models using Python. The NER model identifies entities in text, while the image classification model categorizes images into predefined classes.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [NER Model](#ner-model)
  - [Image Classification Model](#image-classification-model)
- [Models](#models)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Installation
To set up the project, clone the repository and install the required packages:

```bash
git clone https://github.com/DMan113/Task_2-Named-entity-recognition-image-classification-.git
cd ner-image-classification
pip install -r requirements.txt
```

## Usage
### NER Model
To train the NER model, run the following command:

```bash
python train_ner.py
```

To perform inference using the trained NER model:

```bash
python inference_ner.py "Enter your sentence here"
```

### Image Classification Model
To train the image classification model, use:

```bash
python train_image_classifier.py
```

To perform inference on an image:

```bash
python inference_image_classifier.py path/to/image.jpg
```

## Models
- **NER Model**: Utilizes BERT for token classification to identify entities in text.
- **Image Classification Model**: Implements a Convolutional Neural Network (CNN) to classify images into categories.

## Dataset
- **NER Dataset**: A JSON file containing sentences and their corresponding entity labels.
- **Image Dataset**: A structured collection of images categorized into different classes (e.g., animals, objects, etc.).

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is created for educational purposes and is not intended for commercial use.