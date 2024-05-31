from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.utils import img_to_array, load_img
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import os

app = Flask(__name__)

# Define available models


def get_pretrained_model(model_name):
    if model_name == 'VGG16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Identity()
    elif model_name == 'VGG19':
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Identity()
    elif model_name == 'AlexNet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Identity()
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
    else:
        raise ValueError('Unsupported model name')
    return model


# Load the pretrained models
available_models = {
    'VGG16': get_pretrained_model('VGG16'),
    'VGG19': get_pretrained_model('VGG19'),
    'AlexNet': get_pretrained_model('AlexNet'),
    'ResNet50': get_pretrained_model('ResNet50')
}

# Load the trained top models
top_models = {
    'VGG16': load_model('models/savevgg16.h5'),
    'VGG19': load_model('models/savevgg19.h5'),
    'AlexNet': load_model('models/savealexnet.h5'),
    'ResNet50': load_model('models/saveresnet50.h5')
}

# Define transforms for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the animals
animals = ['bear', 'cougar', 'coyote', 'cow', 'crocodiles', 'deer', 'elephant', 'giraffe', 'goat',
           'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penguin', 'sheep', 'skunk', 'tiger', 'zebra']


def read_image(path):
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.reshape((1,) + image.shape)  # Add batch dimension
    return image


def test_single_image(path, model_name):
    # Read and preprocess the image
    image = read_image(path)

    # Convert image to torch tensor
    image = torch.tensor(image, dtype=torch.float32)
    # Change dimension order to [batch_size, channels, height, width]
    image = image.permute(0, 3, 1, 2)

    # Load the selected model
    model = available_models[model_name]
    model.eval()

    # Extract features using the pretrained model
    with torch.no_grad():
        features = model(image)

    # Convert features to numpy array
    features = features.detach().numpy()

    # Make predictions using the trained top model
    top_model = top_models[model_name]
    preds = top_model.predict(features)

    # Determine the predicted class
    class_prob = list(preds[0])
    max_prob = max(class_prob)
    pred_class = class_prob.index(max_prob)
    pred_label = animals[pred_class]

    return pred_label


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']
        model_name = request.form['model']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', file.filename)
        file.save(file_path)

        # Make prediction
        pred_label = test_single_image(file_path, model_name)
        return jsonify({'pred_label': pred_label})
    return render_template('index.html', models=available_models.keys())


if __name__ == '__main__':
    app.run(debug=True)
