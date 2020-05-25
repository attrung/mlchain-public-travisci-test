## Introduction

MLChain class Client allows you to pass your Machien Learning model's output seamlessly between different 
computers, servers, and so on.

The below example uses MLChain client to make request from http://localhost:5000, that is hosted by ourself. 
In real examples, this can be any website url which contain our model.

## Tutorial

First, follow the tutorial [here](../Model Deployment/tutorial.md) and deploy your model at http://localhost:5000, if you haven't already done so.

Next, create a file <b> client.py </b> any where on your computer. At the same time,
download <a href="https://drive.google.com/u/6/uc?id=15wqHzVhFzbusivB7eHB0jWHlA1CIE-DF&export=download" target="_blank"> <b> this </b> </a> image to that folder.

In that file, include the following code:

```python
from mlchain.client import Client
import cv2

image_name = '19.png' # downloaded image

if __name__ == '__main__':
    # tell the system to use the model currently hosting in localhost, port 5000
    model = Client(api_address='localhost:5000',serializer='json').model(check_status=False)
    
    # import our image
    img = cv2.imread('19.png')
    
    # get model response on image
    res = model.image_predict(img)
    
    # print result
    print(res)
```

In your terminal, running 

    $ python client.py

will return "res" as the model's response. 

In software development, we believe using this service allows programmers to better transfer AI models' final results and allowing 
for more cooperation between programmers. This allows you to communicate without having to build complex API systems in the company.


# TODO:
Đoạn này em (Trung) đang k hiểu nó đang hoạt động thế nào ạ. Mai anh bảo em nhé.
### Sending requests to the REST API
Now you can send request to this API.

```bash
curl --location --request POST 'http://localhost:5000/call/predict' --form 'input=ok'
```

for