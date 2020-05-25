import weakref
from mlchain.storage import Path
import uuid
import os
from mlchain.log import format_exc, except_handler
from mlchain.observe.apm import get_transaction
import time


class RemoteFunction:
    def __init__(self, client, url, name):
        """
        Remote Function Call
        :client: Client to communicate, which can not be None
        :url: url to call 
        """
        assert client is not None

        self.client = client
        self.url = url
        self.is_async = False
        self.__name__ = name

    def to_async(self):
        return AsyncRemoteFunction(self.client, self.url, self.__name__)

    def __call__(self, *args, **kwargs):
        args = list(args)
        files_args = {}
        files = []
        # Process files in args
        for idx, item in enumerate(args):
            if isinstance(item, Path):
                new_file_name = str(uuid.uuid4())

                args[idx] = ""
                files.append((new_file_name, (os.path.split(item)[1], open(item, 'rb'), 'application/octet-stream')))
                files_args[new_file_name] = idx
            elif isinstance(item, list) and all([isinstance(x, Path) for x in item]):
                for sub_idx, sub_item in enumerate(item):
                    new_file_name = str(uuid.uuid4())
                    item[sub_idx] = ""

                    files.append(
                        (new_file_name, (os.path.split(sub_item)[1], open(sub_item, 'rb'), 'application/octet-stream')))
                    files_args[new_file_name] = (idx, sub_idx)

        # Process files in kwargs
        drop_key = []
        for key, item in kwargs.items():
            if isinstance(item, Path):
                kwargs[key] = ""
                files.append((key, (os.path.split(item)[1], open(item, 'rb'), 'application/octet-stream')))
                drop_key.append(key)
            elif isinstance(item, list) and all([isinstance(x, Path) for x in item]):
                for sub_idx, sub_item in enumerate(item):
                    item[sub_idx] = ""

                    files.append((key, (os.path.split(sub_item)[1], open(sub_item, 'rb'), 'application/octet-stream')))
                drop_key.append(key)

        for key in drop_key:
            kwargs.pop(key)

        input = {
            'input': (tuple(args), kwargs),
            'files_args': files_args
        }
        headers = {}
        transaction = get_transaction()
        if transaction:
            headers['Traceparent'] = transaction.trace_parent.to_string()
        output = self.client.post(url=self.url, input=input, files=files, headers=headers)

        if 'error' in output:
            with except_handler():
                raise AssertionError("\nREMOTE API ERROR: {0}".format(output['error']))
        else:
            return output['output']


class AsyncRemoteFunction(RemoteFunction):
    def __init__(self, client, url, name):
        """
        Async Remote Function Call
        :client: Client to communicate, which can not be None
        :url: url to call 
        """
        RemoteFunction.__init__(self, client, url, name)
        self.is_async = True

    def to_sync(self):
        return RemoteFunction(self.client, self.url, self.__name__)

    async def __call__(self, *args, **kwargs):
        return RemoteFunction.__call__(self, *args, **kwargs)


class AsyncStorage:
    def __init__(self, function):
        self.function = function

    def get(self, key):
        return AsyncResult(self.function(key))

    def get_wait_until_done(self, key, timeout=100, interval=0.5):
        start = time.time()
        result = AsyncResult(self.function(key))
        while not (result.is_success() or time.time() - start > timeout):
            time.sleep(interval)
            result = AsyncResult(self.function(key))
        return result


class AsyncResult:
    def __init__(self, response):
        self.response = response

    @property
    def output(self):
        if 'output' in self.response:
            return self.response['output']
        else:
            return None

    @property
    def status(self):
        if 'status' in self.response:
            return self.response['status']
        else:
            return None

    @property
    def time(self):
        if 'time' in self.response:
            return self.response['time']
        else:
            return 0

    def is_success(self):
        if self.status == 'SUCCESS':
            return True
        else:
            return False

    def json(self):
        return self.response


class MlchainModel:
    """
    Mlchain Client Model Class
    """

    def __init__(self, client, name, version='lastest', check_status=True):
        """
        Remote model  
        :client: Client to communicate, which can not be None
        :name: Name of model 
        :version: Version of model 
        :check_status: Check model is exist or not, and get description of model 
        """
        assert client is not None and isinstance(name, str) and isinstance(version, str)

        self.client = client
        self.name = name
        self.version = version

        if len(self.name) == 0:
            # Use original API addresss
            self.pre_url = ""
        else:
            # Specific name and version 
            self.pre_url = "{0}/{1}/".format(self.name, self.version)

        self.all_func_des = None
        self.all_func_params = None

        if check_status:
            output_description = self.client.get('{0}api/description'.format(self.pre_url))
            if 'error' in output_description:
                with except_handler():
                    raise AssertionError("ERROR: Model {0} in version {1} is not found".format(name, version))
            else:
                output_description = output_description['output']
                self.__doc__ = output_description['__main__']
                self.all_func_des = output_description['all_func_des']
                self.all_func_params = output_description['all_func_params']
                self.all_attributes = output_description['all_attributes']

        self._cache = weakref.WeakValueDictionary()
        self.store_ = None

    @property
    def store(self):
        if self.store_ is None:
            self.store_ = AsyncStorage(
                RemoteFunction(client=self.client, url='{0}call/{1}'.format(self.pre_url, 'store_get'),
                               name='store_get'))
        return self.store_

    def __check_function(self, name):
        if self.all_func_des is not None:
            if name in self.all_func_des:
                return True
            else:
                return False
        else:
            return True

    def __check_attribute(self, name):
        if self.all_attributes is not None:
            if name in self.all_attributes:
                return True
            else:
                return False
        else:
            return True

    def __getattr__(self, name):
        if name in self._cache:
            true_function = self._cache[name]
        else:
            if not self.__check_function(name):
                if not self.__check_attribute(name) and not name.endswith('_async'):
                    with except_handler():
                        raise AssertionError("This model has no method or attribute name = {0} or it hasnt been served. The only served is: \n\
                                            Functions: {1} \n\
                                            Attributes: {2}".format(name, list(self.all_func_des.keys()),
                                                                    list(self.all_attributes)))
                else:
                    return RemoteFunction(client=self.client, url='{0}call/{1}'.format(self.pre_url, name), name=name)()
            else:
                true_function = RemoteFunction(client=self.client, url='{0}call/{1}'.format(self.pre_url, name),
                                               name=name)
                self._cache[name] = true_function

        return true_function

    def __eq__(self, other):
        return self.client is other.client and self.name == other.name and self.version == other.version

    def __hash__(self):
        return hash(self.client) + hash(self.name) + hash(self.version)


class AsyncMlchainModel:
    """
    Mlchain Client Model Class
    """

    def __init__(self, client, name, version='lastest', check_status=True):
        """
        Remote model  
        :client: Client to communicate, which can not be None
        :name: Name of model 
        :version: Version of model 
        :check_status: Check model is exist or not, and get description of model 
        """
        assert client is not None and isinstance(name, str) and isinstance(version, str)

        self.client = client
        self.name = name
        self.version = version

        if len(self.name) == 0:
            # Use original API addresss
            self.pre_url = ""
        else:
            # Specific name and version 
            self.pre_url = "{0}/{1}/".format(self.name, self.version)

        self.all_func_des = None
        self.all_func_params = None

        if check_status:
            output_description = self.client.get('{0}api/description'.format(self.pre_url))
            if 'error' in output_description:
                with except_handler():
                    raise AssertionError("ERROR: Model {0} in version {1} is not found".format(name, version))
            else:
                output_description = output_description['output']
                self.__doc__ = output_description['__main__']
                self.all_func_des = output_description['all_func_des']
                self.all_func_params = output_description['all_func_params']
                self.all_attributes = output_description['all_attributes']

        self._cache = weakref.WeakValueDictionary()

    def __check_function(self, name):
        if self.all_func_des is not None:
            if name in self.all_func_des:
                return True
            else:
                return False
        else:
            return True

    def __check_attribute(self, name):
        if self.all_attributes is not None:
            if name in self.all_attributes:
                return True
            else:
                return False
        else:
            return True

    def __getattr__(self, name):
        if name in self._cache:
            true_function = self._cache[name]
        else:
            if not self.__check_function(name):
                if not self.__check_attribute(name) and not name.endswith('_async'):
                    with except_handler():
                        raise AssertionError("This model has no method or attribute name = {0}".format(name))
                else:
                    return RemoteFunction(client=self.client, url='{0}call/{1}'.format(self.pre_url, name), name=name)()
            else:
                true_function = AsyncRemoteFunction(client=self.client, url='{0}call/{1}'.format(self.pre_url, name),
                                                    name=name)
            self._cache[name] = true_function

        return true_function

    def __eq__(self, other):
        return self.client is other.client and self.name == other.name and self.version == other.version

    def __hash__(self):
        return hash(self.client) + hash(self.name) + hash(self.version)
