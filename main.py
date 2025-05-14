import argparse
import cv2
import insightface
from insightface.model_zoo.inswapper import INSwapper
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

def load_image(image_path):
    """Loads an image using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img

def detect_faces(image, app):
    """Detects faces using InsightFace."""
    faces = app.get(image)
    if not faces:
        raise ValueError("No faces detected in the image.")
    return faces

def swap_faces(target_img, source_face, target_faces, inswapper_model: INSwapper):
    """
    Swaps faces using the INSwapper model from InsightFace.
    """
    # Perform face swapping using the INSwapper model
    swapped_img = target_img
    if FACE_NUMBER == 0:
        for target_face in target_faces:
            swapped_img = inswapper_model.get(img=swapped_img, target_face=target_face, source_face=source_face)
    elif len(target_faces) < FACE_NUMBER:
        print(f"Warning: face_number {face_number} exceeds number of detected faces ({len(target_faces)}). No face swapped.")
    else:
        swapped_img = inswapper_model.get(img=swapped_img, target_face=target_faces[FACE_NUMBER-1], source_face=source_face)

    return swapped_img

def face_swap(source_image_path, target_image_path, model_path, output_path="swapped_image.jpg"):
    """
    Performs face swapping using InsightFace for detection and INSwapper for swapping.
    """
    print("Initializing InsightFace detector...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Detector initialized.")

    print("Loading INSwapper model...")
    inswapper_model = insightface.model_zoo.get_model(model_path)
    print("INSwapper model loaded.")

    # Load source and target images
    print(f"Loading source image: {source_image_path}")
    source_img = load_image(source_image_path)

    print(f"Loading target image: {target_image_path}")
    target_img = load_image(target_image_path)

    # Detect faces
    print("Detecting faces in source image...")
    source_faces = detect_faces(source_img, app)

    print("Detecting faces in target image...")
    target_faces = detect_faces(target_img, app)

    # Use the first detected face for swapping
    source_face = source_faces[0]

    # Perform face swapping
    print("Swapping faces...")
    swapped_img = swap_faces(target_img, source_face, target_faces, inswapper_model)

    # Save and display the result
    print(f"Saving swapped image to {output_path}")
    cv2.imwrite(output_path, swapped_img)

    # Display the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    plt.title("Source Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    plt.title("Target Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(swapped_img, cv2.COLOR_BGR2RGB))
    plt.title("Swapped Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Face swapping complete.")

if __name__ == "__main__":
    source_image_path = "source.jpg"
    target_image_path = "target.jpg"
    model_path = "./models/inswapper_128.onnx"
    output_image_path = "swapped_image.jpg"

    parser = argparse.ArgumentParser(description="Swap a face using INSwapper")
    parser.add_argument("face", type=int, nargs='?', help="Face number to swap (0 = all faces)", default=0)
    args = parser.parse_args()

    FACE_NUMBER = args.face

    try:
        face_swap(source_image_path, target_image_path, model_path, output_image_path)
    except Exception as e:
        print(f"Error: {e}")

