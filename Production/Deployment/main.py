from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from imagePrediction import ImagePredictor

from PIL import Image

result     = "/Users/narekgeghamyan/Classes/MLE_bootcamp/Capstone_Project/Production/Deployment/Results/segmented_image.png"

app = FastAPI()

predictor = ImagePredictor()

@app.post("/scorefile/")
def create_upload_file(file: UploadFile = File(...)):
    image = Image.open(file.file)
    print("Image converted to PIL image..")
    image_size = image.size
    image = image.resize((int(image.size[0]*0.25), int(image.size[1]*0.25)), Image.ANTIALIAS)
    print("Creating segmented image. This will take a minute...")
    predictor.make_prediction(image)
    print("Segmented image created!")
    return(FileResponse(result))