# Tutorial

Build your first Machine Learning app using MLchain. I this quickstart tutorial we will
use MLchain to build our first ML chain app. First, clone the quick-start repository [here]().

In this repository, you can find the following files:

```bash    
    /root
        -> model.pt
        -> app.py    
```

The <b> model.pt </b> file contains the model we already trained using the MNIST dataset and Pytorch.
 
The <b> app.py </b> file is where we will use to deploy our model.

Upon opening the app.py files, we will import the following libraries:

```python
# pytorch dependencies
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# support for image processing
from PIL import Image
import numpy as np

# mlchain libraries
from mlchain.base import ServeModel
```

Next, redefine our model. This model should be defined the same as your prior model for training.

```python
# redefine our model
class Net(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
```

Next, create a model instance. This will be our main working area. 

```python
class Model():
    # define and load our prior model
    def __init__(self):
        ...

    def image_predict(self, img:np.ndarray):
        ...

```

To begin, we download and import the current state of our model. This should be done under the <b> init() </b> function.

```python
    # define our model
    self.model = Net()

    # load model state_dict
    self.model.load_state_dict(torch.load('model.pt'))

    # set model in evaluation mode
    self.model.eval()
```

We also define <b> transform </b> to transform any input images that are uploaded under the init function.

```python
    self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                         transforms.Resize(28),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
```

Next, for our <b> image_predict() </b> function, we simply take an image as the input and return the corresponding
prediction of our model. This function start with turning our img input (originaly a numpy array) into 
a PIL.Image instance. This helps our transform function defined by pytorch transform the image for our analysis.

We also reshape our image into 4 dimensions tensor as the first value represents the batch_size, which is 1.
```python
    def image_predict(self, img:np.ndarray):

        # form an PIL instance from Image
        img = Image.fromarray(img)
    
        # transform image using our defined transformation
        img = self.transform(img)
    
        # reshape image into 4 - dimensions
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
```       

We can now add prediction and return our final result under <b> image_predict() </b>:
```python
    # predict class using our model
    with torch.no_grad():
        # forward function
        preds = self.model(img)

        # get maximun value
        pred = np.argmax(preds, axis=1)

    # return our final result (predicted number)
    return int(pred)
```

To define and return our final result, we use MLchain's provided functions 
including serve_model and FlaskServer to host our app.

```python
# deploying our model
# define model
model = Model()

# serve model
serve_model = ServeModel(model)

# run model using flask
if __name__ == '__main__':
    from mlchain.rpc.server.flask_server import FlaskServer
    FlaskServer(serve_model).run(port=5000, threads = 5)