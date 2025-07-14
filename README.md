# GuideCAD: A lightweight multimodal framework for 3D CAD model generation via prefix embedding

## Create a Virtual Environment
All experiments were conducted on Ubuntu 22.04.5 LTS. To set up the required environment, run:
```
conda env create --file environment.yaml
```

## Dataset Preparation and Preprocessing
Please refer to the DeepCAD repository for the primary dataset. After downloading, extract the contents into the `dataset/data` directory. Additionally, the dataset we constructed for GuideCAD can be downloaded from this [link](https://www.dropbox.com/home/GuideCAD%20Dataset).

### Preprocess CAD Images and Embeddings
If you wish to generate 3D CAD images and preprocess them into image embeddings using a pretrained image encoder for improved training efficiency, execute the following commands:
```
cd dataset
python3 generate_cad_image.py
python3 generate_image_embedding.py
```

### Generating Text Prompts
To convert a CAD sequence into Command Lines text, use the vector_to_text function. Subsequently, the generate_cad_description function can be employed to generate prompts corresponding to each command and its parameters. Both functions are implemented in the dataset/generate_prompt.py script.


## Training
To train GuideCAD, text prompts are pre-tokenized and stored as token sequences, while images are similarly converted into image embeddings prior to training. Use the following command to start the training process:
```
python3 train.py --config configs/guidecad.yaml
```

## Evaluation
For evaluation, GuideCAD supports various metrics including Coverage (COV), Minimum Matching Distance (MMD), and Jensen–Shannon Divergence (JSD).
Before evaluation, you must sample point clouds from both the generated CAD models (derived from predicted CAD sequences) and the ground truth CAD models. Run the following scripts to evaluate all metrics.
```
cd evaluator
python3 evaluate_acc_f1.py --src "YOUR_PREDICTED_CAD_SEQUENCE_FOLDER" # ex. checkpoints/guidecad/h5
python3 evaluate_cd.py --src "YOUR_PREDICTED_CAD_SEQUENCE_FOLDER"     # ex. checkpoints/guidecad/h5
python3 evaluate_gen.py --src "YOUR_PREDICTED_CAD_SEQUENCE_FOLDER"    # ex. checkpoints/guidecad/h5_pc
```