import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.transform import resize
import cv2
from time import sleep
# Load the trained model
model_path = 'land.h5'
model = load_model(model_path)

# Define the path to the test image
image_path = 'test_images/18.png'

image_path1 = image_path
font = cv2.FONT_HERSHEY_SIMPLEX
# Preprocess the image and resize it
img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.
img=cv2.cvtColor(img_array[0], cv2.COLOR_RGB2BGR)
img= cv2.resize(img, (500, 350))
kim=img
cv2.putText(kim,"Input", (20, 40), font , 0.75, (0, 0, 255), 2)
cv2.imshow('input', img)
cv2.waitKey(2000)

def detect_major_color(image_path):
    # Read the image
    image1 = cv2.imread(image_path)
    
    # Convert image to RGB 
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Convert to float type
    pixels = np.float32(pixels)
    
    # Define criteria, number of clusters (K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 2  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    
    # Reshape labels to match the shape of the image
    labels = labels.reshape(image.shape[:2])
    
    # Find count of pixels for each cluster
    counts = np.bincount(labels.flatten())
    
    # Find the color with the most pixels
    major_color_label = np.argmax(counts)
    major_color = centers[major_color_label]
    plant=""
    
    # Check if the major color falls within the specified RGB range
    if (80 <= major_color[0] <= 110) and (60 <= major_color[1] <= 90) and (60 <= major_color[2] <= 90):
        plant="Red Spinach"
    if (68 <= major_color[0] <=115 ) and (102 <= major_color[1] <= 130) and (85 <= major_color[2] <= 110):
        plant="Green Wheat"
    if (150 <= major_color[0] <=190 ) and (165 <= major_color[1] <= 200) and (140 <= major_color[2] <= 170):
        plant="Tea"
    if (130 <= major_color[0] <=140 ) and (130 <= major_color[1] <= 140) and (110 <= major_color[2] <= 125):
        plant="Orange"
    
    # Draw boundary around the major color
    mask = np.zeros_like(image)
    mask[labels == major_color_label] = image[labels == major_color_label]
    contours, _ = cv2.findContours((labels == major_color_label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image1, contours, -1, (255, 0, 0), thickness=2)
    
    # Resize the boolean mask to match the dimensions of the image
    resized_labels = cv2.resize((labels == major_color_label).astype(np.uint8), (image.shape[1], image.shape[0]))
    mask = np.zeros_like(image)
    mask[resized_labels == 1] = (255, 165, 0)
    
    # Apply the mask to the original image
    masked_image = cv2.addWeighted(image1, 1, mask, 0.5, 0)
    
    # Convert major_color to RGB format
    major_color_rgb = tuple(major_color)
    print(major_color_rgb)
    
    # Calculate the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    
    # Count the number of non-zero pixels in the mask
    masked_pixels = np.count_nonzero(np.any(mask != 0, axis=-1))
    
    # Calculate the percentage of the masked area
    masked_percentage = int((masked_pixels / total_pixels) * 100)
    text = "Agricultural Area in this image:"+str(masked_percentage)+"%"
    print("Percentage of masked area:", masked_percentage," %")
    masked_image= cv2.resize(masked_image, (500, 350))
    cv2.putText(masked_image, text, (20, 60), font , 0.75, (128, 64, 64), 2)
    cv2.putText(masked_image, "detected plant : "+plant , (20, 100), font , 0.75, (128, 64, 64), 2)
    cv2.imshow('Final Detection', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite('final_detection.png',cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    return plant


# Make predictions
predictions = model.predict(img_array)
print(predictions)
predicted_class_index = np.argmax(predictions)
confidence = predictions[0][predicted_class_index]

# Threshold the predicted probabilities
threshold = 0.5  
predicted_class_binary = (predictions > threshold).astype(int)

# Segment the image based on the predicted classes
segmented_image = label(predicted_class_binary)


print(predicted_class_binary)
# Resize segmented image to match the dimensions of the input image array
segmented_image_resized = resize(segmented_image, img_array[0].shape[:2], order=0, anti_aliasing=False)
contours, _ = cv2.findContours(segmented_image_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_image = cv2.cvtColor(img_array[0], cv2.COLOR_RGB2BGR)
cv2.drawContours(boundary_image, contours, -1, (0, 255, 0), 2)
boundary_image = cv2.resize(boundary_image, (500, 350))
# Convert the image to grayscale
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image=cv2.resize(gray_image, (500, 350))
_, segmenteimage = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.putText(segmenteimage,"segmented_image", (20, 40), font , 0.75, (0, 0, 255), 2)
cv2.imshow('segmented_image', segmenteimage)
cv2.waitKey(2000)
kim=boundary_image
cv2.putText(kim,"boundary_image", (20, 40), font , 0.75, (0, 0, 255), 2)
cv2.imshow('boundary_image',kim)
cv2.waitKey(2000)
land=""
confident =str(int(confidence*100))
print("confidence = "+confident+"%")
plant=" "
if(predicted_class_index==7):
    land="Forest"
    
elif(predicted_class_index==0):
    land="Agricultural"
    
else:
    land="Other"
text= "Detected land : "+land
cv2.putText(boundary_image, text, (20, 60), font , 0.75, (0, 0, 255), 2)
# Display the image with boundaries
cv2.imshow('First classification', boundary_image)
cv2.imwrite("first_classification.png",boundary_image)
cv2.waitKey(3000)
if(land=="Agricultural"):
    plant = detect_major_color(image_path1)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.resize(image, (500, 350))
    cv2.imshow('BGR2RGB',image)
    cv2.waitKey(2000)
cv2.waitKey(0)
cv2.destroyAllWindows()
