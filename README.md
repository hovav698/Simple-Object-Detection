This repo demonstrate how basic object detection algorithm works. It also shows a fun way to generate simulated dataset instead of using a real data.



**The Model**

The algorithm have two objectives:

Classification: Determine what kind of object is in the image.

Localization: Predict the bounding box coordinates. This is a regression problem:

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123609268-30808880-d808-11eb-83ea-4cd2d8af38c2.png">

We will created VGG16 convolution model with the following outputs:

- 4 outputs for location prediction. The first and second outputs will represent the left coordinate and the buttom coordinate. The third and fourth coordinate will represent the bounding box width and height.
- K outputs for the object class prediction, where K represent the number of classes.
- 1 outputs for the "object appeared" flag. It will predict wheather object is exist in the image.
 

**The Loss Function**

The loss will be composed of the classification and localization loss.

Localization Loss - The coordinates will be normalized coordinates, with a value range between 0 and 1. Therefor we will use the binnary cross entropy loss function. It outputs how far the predicted coordinates are from the target coordinates.

Classification Loss - Categorical cross entopy loss. It outputs how close are the predicted class score to the target class.

Object appeared loss - Binnary Croos entropy loss. Outputs score between 0 and 1 which represent the probability that object appeared. 

If there is no object in the image, we don't want the location and classification loss will be taken into the loss calculation. Therefore the classification and location loss will be multiple by the object appear flag. 

The final loss will be sum of the above losses:

![image](https://user-images.githubusercontent.com/71300410/123615721-3e390c80-d80e-11eb-885e-ddee9d3d2592.png)

Where t and p represent the target and prediction vector.



**Dataset Generation**

For creating the model we need to have dataset that contains images of single object, and the target should contains the location coordinate, the target flag and the appear flag. In this project I decided to generate the dataset since since it's much faster to train it then using external image dataset.

For generaing the dataset I've download images of different animals (in a png format) and backgrounds. To create the target images, i've pasted the animal image to the background in random positions.  
It's important to use a PNG image format since it contain the transparent channel - it's the 4'th channel that contains the mask of the animal. For example, plotting the 4'th channel of the tiger image will output the following mask: 

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123619960-63c81500-d812-11eb-8330-ba75c52f301e.png">


The data generation process goes as follow:

- Choose random background
- Randomly choose if the object appear flag. If object appeard, randomly choose animal from the data.
- Randomly flip and resize the animal image to add data augmentation. 
- Choose random coordinates on the background image - this is where the animal image will be paste.
- Take a slice of the background for those coordinate. 
- Multiple the background slice with the mask. It will change the pixel value to zero in the location where the animal is located, and won't be affected it other parts because the mask value is 1 in those areas:

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123622760-4e081f00-d815-11eb-95d4-4164812f4992.png">

- Add the first 3 channel of the animal image to the bacgrkound slice. Now the background slice will contain the animal only in the previously masked area:
  
<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123623005-a3dcc700-d815-11eb-979b-19b17028b402.png">

- Place the edited background slice back to the original position. This is the final image result:

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123623551-3c734700-d816-11eb-9727-6e14fa802eff.png">

- Create the target vector - object location, object class and the appear flag. 

- Create data generatrion function that outputs batch of images and targets. It will be used for the model training.

Few examples of the random generated images:

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123592637-29507f00-d7f6-11eb-8e7e-b10ce968af46.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123592890-85b39e80-d7f6-11eb-81e7-19632bffcb49.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123593129-d62afc00-d7f6-11eb-8f74-8c55245c6a55.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123593401-22763c00-d7f7-11eb-9897-1e9770cb8059.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123593919-b8aa6200-d7f7-11eb-9de0-d7bd317f60c2.png">
<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123594578-8d744280-d7f8-11eb-8bcb-c8436d69b415.png">



**Results**

The loss per Epoch:

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123625781-ca503180-d818-11eb-9f72-31cbbcb1495b.png">

Now that the model has learned to detect object, we can randmly generate images, feed it to the model and get the result. We will plot a bounding box with the predicted coordinates and will choose the predicted class according the the argmax of the classes scores. Some different examples of the final results:

<img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123628041-5a8f7600-d81b-11eb-9b26-3435322fe7de.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123628251-8ca0d800-d81b-11eb-85ce-4be93b0f36aa.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123628418-bc4fe000-d81b-11eb-9f06-36dedf95c748.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123628782-28324880-d81c-11eb-9b5e-da2d676af202.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123628893-4730da80-d81c-11eb-818c-c7dd388be569.png"><img width="280" alt="RTST" src="https://user-images.githubusercontent.com/71300410/123628973-5e6fc800-d81c-11eb-9d43-69fb5040821a.png">



**Summary**

We were able to get a model that works pretty good for the generated dataset. However, this model is basic and works good only for one objects images. Another downside to this model is - the scale of the object it learned to detect is dependent on the data scale. For general models that can detect multiple object at the same image with different object scales we will need to use much more complicated model. I'm working on that kind a model, called SSD. Will update (-:






