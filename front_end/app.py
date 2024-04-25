from flask import Flask, render_template, url_for, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask import  request
import os
from flask import Flask, render_template
from joblib import load
import torch
from cv2 import imread
import numpy as np
import os
import random
import sys

# Add the directory containing 'src' folder to the Python path
#sys.path.append(r'C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master')
sys.path.append(r'C:\Users\preet\Desktop\ADLDIFID')
# Now you can import the CNN class
from src.cnn.cnn import CNN
from src.feature_fusion.feature_vector_generation import get_patch_yi
app = Flask(__name__)

# Load the pretrained CNN with the CASIA2 dataset
with torch.no_grad():
    our_cnn = CNN()
    our_cnn.load_state_dict(torch.load('../data/output/pre_trained_cnn/CASIA2_WithRot_LR001_b128_nodrop.pt',
                                       map_location=lambda storage, loc: storage))
    our_cnn.eval()
    our_cnn = our_cnn.double()

# Load the pretrained svm model
svm_model = load('../data/output/pre_trained_svm/CASIA2_WithRot_LR001_b128_nodrop.pt')

def get_feature_vector(image_path: str, model):
    feature_vector = np.empty((1, 400))
    feature_vector[0, :] = get_patch_yi(model, imread(image_path))
    return feature_vector

@app.route('/run_single_test/<filename>')
def run_single_test(filename):
    # Define the paths
    upload_folder = app.config['UPLOAD_FOLDER']


    # Construct the dynamic non_tampered_path using the upload folder and filename
    non_tampered_path = os.path.join(upload_folder, filename)
#    non_tampered_path = r'C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master\data\CASIA2\Au'
    #tampered_path = r'C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master\data\CASIA2\Tp'
    print('hio',non_tampered_path)
    # Get a list of files in each directory
    #non_tampered_files = os.listdir(non_tampered_path)
    #tampered_files = os.listdir(tampered_path)

    # Randomly select one image from each directory
    #non_tampered_sample = random.choice(non_tampered_files)
    #tampered_sample = random.choice(tampered_files)

    # Generate feature vectors and predictions for the selected images
    #non_tampered_image_path = os.path.join(non_tampered_path, non_tampered_sample)
    #tampered_image_path = os.path.join(tampered_path, tampered_sample)

    non_tampered_feature_vector = get_feature_vector(non_tampered_path, our_cnn)
    #tampered_feature_vector = get_feature_vector(tampered_image_path, our_cnn)

    non_tampered_prediction = svm_model.predict(non_tampered_feature_vector)
    #tampered_prediction = svm_model.predict(tampered_feature_vector)
    if non_tampered_prediction == 0:
        prediction= 'The result  is ' \
                    'Authentic with Prediction[0]'
    else:
        prediction = "The result is" \
                     "Tampered with Prediction[1]"
    # Render a template with the predictions
    return render_template('single_test_output.html',
                           filename=filename,
                           image_path=non_tampered_path,
                           non_tampered_image=non_tampered_path,

                           non_tampered_prediction=prediction
                           )

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(image_path)

                # Redirect to single_test_output route with image filename as a parameter
                return redirect(url_for('single_test_output', filename=image.filename))

    return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

#UPLOAD_FOLDER = "C:\\Users\\preet\\PycharmProjects\\pythonProject1\\flask front end\\uploads"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/upload', methods=['POST'])
def upload_image():
    print("hi")
    logger.debug("Hiii")
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']
    logger.debug("Hiii")
    if image.filename == '':
        return redirect(request.url)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    logger.debug("Hiii")
    image.save(image_path)
    #return redirect(url_for('display', filename=image.filename))
    return redirect(url_for('run_single_test', filename=image.filename,image_path=image_path))
    # return 'Image successfully uploaded!',image_path

@app.route('/single_test_output/<filename>')
def single_test_output(filename):
    # Generate feature vector and prediction for the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    feature_vector = get_feature_vector(image_path, our_cnn)
    prediction = svm_model.predict(feature_vector)

    # Render the single_test_output.html template with the prediction and image filename
    return render_template('single_test_output.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)


#cd .\front_end\      python .\app.py