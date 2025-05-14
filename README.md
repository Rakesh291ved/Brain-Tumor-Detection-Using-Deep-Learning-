# Brain-Tumor-Detection-Using-Deep-Learning-
Brain Tumor Detection Using Deep Learning 

🧠 Brain Tumor Detection Using Deep Learning 🤖
This project involves detecting brain tumors using deep learning and image classification with a ResNet50 model. The model classifies brain tumor images into four categories: Meningioma, Glioma, Pituitary, and None. The system uses a Flask web app to allow users to upload medical images, and the model predicts the tumor type and provides a diagnosis.

🌟 Project Overview
The Brain Tumor Detection system is a web-based application powered by PyTorch and Flask. It uses a ResNet50 model pre-trained on brain tumor images and fine-tuned for four specific tumor types. Once an image is uploaded, the app makes a prediction and returns the tumor classification.

🚀 Features
🔍 Real-time Brain Tumor Classification: Upload medical images to get an instant prediction on the tumor type.

🖼 Image Preprocessing: Handles different image formats (png, jpg, jpeg, gif) and resizes them for input to the model.

📧 Model Deployment: Runs in a lightweight Flask app with PyTorch as the backend model.

🧠 Deep Learning Model: Based on ResNet50, a pre-trained CNN that is fine-tuned to classify brain tumor images.

🛠️ Tech Stack
Component	Technology
Backend	Python, Flask, PyTorch
Model	ResNet50 (fine-tuned for brain tumors)
Frontend	HTML, CSS (Flask Templates)
Email Service	None
Deployment	Local, can be deployed on cloud servers

🧑‍💻 Model Architecture
Pretrained Model: The app uses a ResNet50 architecture, a deep convolutional neural network, pretrained on general image datasets. The last layer of the model has been modified for 4 classes:

None

Meningioma

Glioma

Pituitary

The last fully connected layers are replaced with layers suitable for the task at hand, and the model is fine-tuned with a dataset of brain tumor images.

Brain-Tumor-Detection/
│
├── app.py                       # Main Flask application with the prediction route
├── static/
│   └── photos/                  # Upload folder for image files
├── templates/
│   ├── DiseaseDet.html          # Homepage with instructions
│   ├── uimg.html                # Page for image upload
│   └── pred.html                # Page to show prediction result
├── models/
│   └── bt_resnet50_model.pt     # Pre-trained and fine-tuned model file
├── brain_tumor_dataset/         # Image dataset (training)
└── README.md                    # This file

💡 How It Works
User Uploads Image: The user visits the web page, uploads a brain MRI or CT scan image.

Image Preprocessing: The uploaded image is resized and transformed to match the model input.

Model Prediction: The model makes a prediction based on the image and outputs one of the four classes:

Meningioma

Glioma

Pituitary

None

Result Display: The predicted tumor type is displayed on a results page.

📧 How to Use the Flask Web App
Step 1: Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
Step 2: Install Dependencies
Install the required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
Make sure you have PyTorch installed with GPU support if available.

Step 3: Run the Flask Application
Run the application locally:

bash
Copy
Edit
python app.py
This will start the Flask app at http://127.0.0.1:5000/. You can access the web interface from there.

Step 4: Upload an Image
On the homepage, upload a valid brain tumor image in the formats: png, jpg, jpeg, gif.

The model will process the image and display the tumor classification result.

📸 Screenshot

⚙️ Model Training (For Developers)
If you want to train the model yourself, here are the steps:

Dataset: The dataset consists of labeled brain tumor images. Make sure to organize the images into respective folders (e.g., Meningioma, Glioma, Pituitary, None).

Training: The training loop uses the ResNet50 architecture, and the model is fine-tuned with an Adam optimizer and negative log-likelihood loss.

Saving the Model: After training, the model is saved in the models/bt_resnet50_model.pt path.

python
Copy
Edit
torch.save(model.state_dict(), model_path)
💾 Model Saving and Loading
Saving the Model:
After training the model, we save it to a .pt file:

python
Copy
Edit
torch.save(model.state_dict(), 'path_to_save_model')
Loading the Model:
When running the web app, we load the model for inference:

python
Copy
Edit
model.load_state_dict(torch.load(model_path))
model.eval()
👨‍💻 Developer
Made with ❤️ by Vedanth Rakesh
Email: vedanthrakesh2910@gmail.com

📃 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute with proper credit.
