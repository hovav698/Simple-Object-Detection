This repo demonstrate how basic object detection algorithm works. 

**How the Algorithm works?**

The algorithm have two goals. Classifythe object, and localize the object.

**Classification**: "This is a monkey"   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  **Localization**: The monkey is located here"

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123606801-f1e9ce80-d805-11eb-9d4e-7166959ce734.png">     &nbsp; &nbsp; &nbsp;  <img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123606546-ad5e3300-d805-11eb-8361-f9b198694737.png">

The classification goal is to tell what kind of object was detected. This is classification problem.

The localization goal is to predict the bounding box coordinates. This is a regression problem.

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123609268-30808880-d808-11eb-83ea-4cd2d8af38c2.png">

We will created VGG16 convolution model with the following outputs:

- 4 outputs for location prediction. The first and second outputs will represent the left coordinate and the buttom coordinate. The third and fourth coordinate will represent the bounding box width and height.
- K outputs for the object class prediction, where K represent the number of classes.
- 1 outputs for the "object appeared" flag. 
 

**The Loss Function**

The loss will be composed of the classification and localization loss.

Localization Loss - The coordinates will be normalized coordinates, with a value range between 0 and 1. Therefor we will use the binnary cross entropy loss function. It outputs how far the predicted coordinates are from the target coordinates.

Classification Loss - Categorical cross entopy loss. It outputs how close are the predicted class score to the target class.

Object appeared loss - Binnary Croos entropy loss. 

If there is no object in the image, we don't want the location and classification will be taken into the loss calculation. Therefore the classification and location loss will be multiple by the object appear flag. 

the final loss will be sum of the above losses:

![image](https://user-images.githubusercontent.com/71300410/123615721-3e390c80-d80e-11eb-885e-ddee9d3d2592.png)

Where t and p represent the target and prediction vector.

**Dataset Generation**

For creating the model we need to have dataset that contains images of single object, and the target should contains the location coordinate, the target flag and the appear flag. In this project I decided to generate the dataset since since it's much faster to train it then using external image dataset.

For generaing the dataset I've download images 

For the purpose of the project I used self generated data


<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123592637-29507f00-d7f6-11eb-8e7e-b10ce968af46.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123592890-85b39e80-d7f6-11eb-81e7-19632bffcb49.png">
<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123593129-d62afc00-d7f6-11eb-8f74-8c55245c6a55.png">
<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123593401-22763c00-d7f7-11eb-9897-1e9770cb8059.png">
<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123593919-b8aa6200-d7f7-11eb-9de0-d7bd317f60c2.png">
<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123594578-8d744280-d7f8-11eb-8bcb-c8436d69b415.png">



