import flask
import torch
from torch import device as DEVICE, load, argmax
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Resize
from io import BytesIO  # Add this import
import os

UPLOAD_FOLDER = os.path.join('static', 'photos')
app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

LABELS = ['None', 'Meningioma', 'Glioma', 'Pituitary']

device = "cuda" if is_available() else "cpu"

# Load the ResNet model
resnet_model = models.resnet50(pretrained=True)
for param in resnet_model.parameters():
    param.requires_grad = False  # Freeze pre-trained weights

# Modify the final layer for your classification (4 classes)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = Sequential(
    Linear(num_ftrs, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 4),
    LogSigmoid()
)

# Load model weights
model_path = 'D:/vs/ML Projects/BRAIN TUMOR DETECTION/models/bt_resnet50_model.pt'
resnet_model.load_state_dict(load(model_path, map_location=DEVICE(device)))

resnet_model.eval()  # Set the model to evaluation mode

def preprocess_image(image_bytes):
    transform = Compose([Resize((512, 512)), ToTensor()])
    img = Image.open(BytesIO(image_bytes))  # Use BytesIO here
    return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = preprocess_image(image_bytes)
    y_hat = resnet_model(tensor.to(device))
    class_id = argmax(y_hat.data, dim=1)
    return str(int(class_id)), LABELS[int(class_id)]

@app.route('/', methods=['GET'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('DiseaseDet.html')

@app.route("/uimg", methods=['GET', 'POST'])
def uimg():
    if flask.request.method == 'GET':
        return flask.render_template('uimg.html')
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(img_bytes)
        return flask.render_template('pred.html', result=class_name, file=file)

@app.errorhandler(500)
def server_error(error):
    return flask.render_template('error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
