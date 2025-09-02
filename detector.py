from pathlib import Path
import face_recognition
import pickle
from collections import Counter
import argparse

from PIL import Image, ImageDraw

# filters out the depricaition warning cause by the removal of the package later in time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

BOUNDING_BOX_COLOR = "red"
TEXT_COLOR = "white"

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="red",
        outline="red",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )



DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test model accuracy with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()

# all three check to see if a directories exists and makes them if they don't
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        # provides a tuple for the cordinates to make a box around the faces
        face_locations = face_recognition.face_locations(image, model=model)
        
        # makes the encoding for the detected faces in the image
            # the encoding are the numeric values which describe the facial features
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # adds the names and their encoding to a specific lists
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # serialize the names and encodings using pickle 
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # test image 
    input_image = face_recognition.load_image_file(image_location)

    # detects inputs faces and their encodings 
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )

    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
 
def validate(model: str = "hog"):
    valid_extensions = (".jpg", ".jpeg", ".png")

    for filepath in Path("validation").rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)