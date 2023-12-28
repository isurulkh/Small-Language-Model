# SmallDisMed: Fine-tuned GPT-2 for Medical Text Generation

## Overview

This repository contains code for training a language model on a dataset of diseases and symptoms. The model is based on the distilgpt2 architecture and is fine-tuned for generating text related to diseases and their symptoms.

SmallDisMed is a fine-tuned GPT-2 model pre-trained on a medical dataset for symptom prediction and text generation tasks. This repository provides the code and resources to train and use the model for generating text related to diseases and symptoms.



## Features:

- Fine-tuned GPT-2 model: Based on the distilgpt2 pre-trained model, optimized for medical language generation.
- Flexible data handling: Accepts customized datasets of disease-symptom pairs for further fine-tuning.
- Text generation: Predicts potential symptoms based on a given disease or generates descriptive text based on disease or symptom prompts.
- Easy to use: Provides well-documented code and instructions for training and generating text.


## Installation

To install the required dependencies, run the following command:

```bash
pip install torch torchtext transformers sentencepiece pandas tqdm datasets
```

## Model Fine-tuneing codes 
```bash
SLM.ipynb
```


## Installation for running Streamlit app

1. **Clone the Repository:**

    ```bash
    https://github.com/isurulkh/Small-Language-Model.git
    cd streamlit-app
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Model Path:**

    Update the Streamlit app code (`app.py`) to include the correct path to your trained language model. Modify the `MODEL_PATH` variable in the code.

## Usage

Run the Streamlit app using the following command:

```bash
streamlit run app.py
```


## Additional Notes:
The model can be further fine-tuned on specific datasets for improved performance.
For advanced usage, refer to the provided code files for detailed implementation details.
Feel free to contribute to the project by raising issues or submitting pull requests!


## Disclaimer:
This model is for research purposes only and should not be used for medical diagnosis or treatment.
Please consult with a qualified healthcare professional for any medical concerns.
