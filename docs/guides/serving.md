You can use MLChain to deploy your models as REST APIs with a single command: `mlchain run server.py`.
### Serve Mode
In order to serve your model, you need to provide an `server.py` file. Inside this file, you need to create a Python class and wrapper by ServeModel.

```python
from mlchain.base import ServeModel

class YourModel:
    def predict(self,input:str):
        '''
        function return input
        '''
        return input

model = YourModel()

serve_model = ServeModel(model)
```

Run command
```bash
mlchain run server.py --host localhost --port 5000
```

### Sending requests to the REST API

Now you can send request to this API.

```bash
curl --location --request POST 'http://localhost:5000/call/predict' --form 'input=ok'
```

Or you can use mlchain Client
```python
from mlchain.client import Client

model = Client(api_address='http://localhost:5000').model(check_status=False)

res = model.predict("ok")
print(res)
```
