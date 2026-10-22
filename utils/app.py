import os
import torch

from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap

from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired

from werkzeug.utils import secure_filename

from torchvision import transforms
from PIL import Image

from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ==========================================
# Flask App
# ==========================================

app = Flask(
    __name__,
    template_folder='../Templates',
    static_folder='../static'
)

app.config['SECRET_KEY'] = 'superscretkey'

app.config['UPLOAD_FOLDER'] = os.path.join(
    app.static_folder,
    'uploads'
)

app.config['ALLOWED_EXTENSIONS'] = {
    'png',
    'jpg',
    'jpeg'
}

Bootstrap(app)

os.makedirs(
    app.config['UPLOAD_FOLDER'],
    exist_ok=True
)


# ==========================================
# Upload Form
# ==========================================

class UploadForm(FlaskForm):

    content = FileField(
        'Content Image',
        validators=[InputRequired()]
    )

    style = FileField(
        'Style Image',
        validators=[InputRequired()]
    )

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        'Alpha',
        default=1.0
    )

    submit = SubmitField(
        'Transfer Style'
    )


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# Load VGG Encoder
# ==========================================

VGG_PATH = 'D:/NST_PROJECT/models/vgg_normalised.pth'

encoder = VGGEncoder(
    VGG_PATH
).to(device)

encoder.eval()


# ==========================================
# Load Trained Decoder
# ==========================================

DECODER_PATH = 'D:/NST_PROJECT/models/decoder.pth'

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device
    )
)

decoder.eval()

print("Trained decoder loaded successfully!")


# ==========================================
# File Validation
# ==========================================

def allowed_file(filename):

    return (
        '.' in filename
        and
        filename.rsplit(
            '.',
            1
        )[1].lower()
        in app.config['ALLOWED_EXTENSIONS']
    )


# ==========================================
# Neural Style Transfer
# ==========================================

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    content_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Convert images to tensors

    content_image = content_transform(
        content_image
    ).unsqueeze(0).to(device)

    style_image = style_transform(
        style_image
    ).unsqueeze(0).to(device)

    # No gradients needed during inference

    with torch.no_grad():

        # Extract VGG features

        content_feats = encoder(
            content_image,
            is_test=True
        )

        style_feats = encoder(
            style_image,
            is_test=True
        )

        # Adaptive Instance Normalization

        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        # Control style strength

        stylized_feats = (
            alpha * stylized_feats
            +
            (1 - alpha) * content_feats
        )

        # Generate stylized image

        stylized_image = decoder(
            stylized_feats
        )

    return stylized_image


# ==========================================
# Save Image
# ==========================================

def save_image(image, path):

    image = image.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(
        0,
        1
    )

    image = transforms.ToPILImage()(image)

    image.save(path)


# ==========================================
# HOME PAGE
# ==========================================

@app.route(
    '/',
    methods=['GET', 'POST']
)
def index():

    form = UploadForm()

    # These are sent to the HTML template

    result_image = None
    content_image = None
    style_image = None
    error = None

    # ======================================
    # Form submitted
    # ======================================

    if form.validate_on_submit():

        # -------------------------------
        # Content image validation
        # -------------------------------

        if not form.content.data:

            error = "Please upload a content image."

        elif not allowed_file(
            form.content.data.filename
        ):

            error = "Invalid content image format."

        # -------------------------------
        # Style image validation
        # -------------------------------

        elif not form.style.data:

            error = "Please upload a style image."

        elif not allowed_file(
            form.style.data.filename
        ):

            error = "Invalid style image format."

        else:

            # ==================================
            # Secure filenames
            # ==================================

            content_filename = secure_filename(
                form.content.data.filename
            )

            style_filename = secure_filename(
                form.style.data.filename
            )

            # ==================================
            # Create upload paths
            # ==================================

            content_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                content_filename
            )

            style_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                style_filename
            )

            # ==================================
            # Save uploaded images
            # ==================================

            form.content.data.save(
                content_path
            )

            form.style.data.save(
                style_path
            )

            try:

                # ==================================
                # Open images
                # ==================================

                content_image_pil = Image.open(
                    content_path
                ).convert("RGB")

                style_image_pil = Image.open(
                    style_path
                ).convert("RGB")

                # ==================================
                # Alpha / style strength
                # ==================================

                alpha = float(
                    form.alpha.data
                )

                alpha = max(
                    0.0,
                    min(
                        1.0,
                        alpha
                    )
                )

                # ==================================
                # Run neural style transfer
                # ==================================

                stylized_image = style_transfer(
                    content_image_pil,
                    style_image_pil,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                # ==================================
                # Output filename
                # ==================================

                result_filename = (
                    'stylized_'
                    + content_filename
                )

                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    result_filename
                )

                # ==================================
                # Save generated image
                # ==================================

                save_image(
                    stylized_image,
                    result_path
                )

                # ==================================
                # IMPORTANT:
                # Give HTML the browser URL
                # ==================================

                result_image = (
                    '/uploads/'
                    + result_filename
                )

                # These are also useful if
                # you want to display them later.

                content_image = (
                    '/uploads/'
                    + content_filename
                )

                style_image = (
                    '/uploads/'
                    + style_filename
                )

                print(
                    "Style transfer completed:"
                )

                print(
                    result_path
                )

                print(
                    "Browser result URL:"
                )

                print(
                    result_image
                )

            except Exception as e:

                error = str(e)

                print(
                    "ERROR:",
                    error
                )

    # ==========================================
    # Send everything to index.html
    # ==========================================

    return render_template(
        'index.html',

        form=form,

        result_image=result_image,

        content_image=content_image,

        style_image=style_image,

        error=error
    )


# ==========================================
# Serve Uploaded Images
# ==========================================

@app.route(
    '/uploads/<filename>'
)
def send_image(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


# ==========================================
# Serve Example Images
# ==========================================

@app.route(
    '/examples/<path:filename>'
)
def send_example(filename):

    return send_from_directory(
        'examples',
        filename
    )


# ==========================================
# Run Flask
# ==========================================

if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False
    )