from mlchain.base.serializer import JsonSerializer, MsgpackSerializer, MsgpackBloscSerializer
from inspect import signature
# from mlchain.log import Logger
# from mlchain.storage.s3 import S3Storage
import logging
# import os
try:
    from gunicorn.app.base import BaseApplication
except Exception as e:
    import warnings
    warnings.warn("Import error {0}".format(e))
    class BaseApplication(object):
        def __init__(self):
            raise ImportError("Can't import gunicorn. Please set gunicorn = False")

class Server:
    def __init__(self, name='mlchain_server',version = '1.0',logger=None):
        # Serializer initalization 
        self.serializers_dict = {
            'application/json': JsonSerializer(),
            'application/msgpack': MsgpackSerializer(),
        }
        try:
            msg_blosc_serializer = MsgpackBloscSerializer()
            self.serializers_dict['application/msgpack_blosc'] = msg_blosc_serializer
        except:
            self.serializers_dict['application/msgpack_blosc'] = self.serializers_dict['application/msgpack']
            warnings.warn("Can't load MsgpackBloscSerializer. Use msgpack instead")
        # Variable default initalization
        self.all_serve_function = set()
        self.all_atrributes = set()
        self.name = name
        self.version = version
        self.__cache__ = '{0}_{1}'.format(name,version)
        self.logger = logger

    def _check_blacklist(self, deny_all_function, blacklist, whitelist):
        output = []
        if deny_all_function:
            for name in dir(self.model):
                attr = getattr(self.model, name)
                
                if not name.startswith("__") and (getattr(attr,"_MLCHAIN_EXCEPT_SERVING",False) or name not in whitelist):
                    output.append(name)
        else:
            for name in dir(self.model):
                attr = getattr(self.model, name)
                
                if not name.startswith("__") and (getattr(attr,"_MLCHAIN_EXCEPT_SERVING",False) or name in blacklist):
                    output.append(name)
        return output

    def _check_all_func(self, blacklist_set):
        """
        Check all available function of class to serve
        """ 

        self.all_serve_function = set()
        for name in dir(self.model):
            attr = getattr(self.model, name)
            
            if (not name.startswith("__") or name == '__call__') and callable(attr) and name not in blacklist_set:
                    self.all_serve_function.add(name)

    def _list_all_atrributes(self):
        return list(self.all_atrributes)

    def _check_all_attribute(self, blacklist_set):
        """
        Check all available function of class to serve
        """ 

        self.all_atrributes = set()
        for name in dir(self.model):
            attr = getattr(self.model, name)
            
            if not name.startswith("__") and not callable(attr):
                if not getattr(attr,"_MLCHAIN_EXCEPT_SERVING",False) and name not in blacklist_set:
                    self.all_atrributes.add(name)

    def _list_all_function(self):
        """
        Get all functions of model
        """
        return list(self.all_serve_function)

    def _list_all_function_and_description(self):
        """
        Get all function and description of all function of model
        """
        output = {}

        for name in self.all_serve_function:
            output[name] = getattr(self.model, name).__doc__
        return output

    def _list_all_function_and_parameters(self):
        """
        Get all function and parameters of all function of model
        """
        output = {}

        for name in self.all_serve_function:
            output[name] = str(signature(getattr(self.model, name)))
        return output

    def _get_description_of_func(self, function_name):
        """
        Get description of a specific function 
        """
        if function_name is None or len(function_name) == 0 or function_name not in self.all_serve_function:
            return "No description for unknown function"

        return getattr(self.model, function_name).__doc__

    def _get_parameters_of_func(self, function_name):
        """
        Get all parameters of a specific function 
        """
        if function_name is None or len(function_name) == 0 or function_name not in self.all_serve_function:
            return "No parameters for unknown function"

        return str(signature(getattr(self.model, function_name)))
    
    def _get_all_description(self):
        """
        Get all description of model
        """
        output = {
            '__main__': self.model.__doc__,
            'all_func_des': self._list_all_function_and_description(),
            'all_func_params': self._list_all_function_and_parameters(),
            'all_attributes': self._list_all_atrributes()
        }
        return output
        
    def _check_status(self):
        """
        Check status of a served model 
        """
        return "pong"

    def run(self):
        """ 
        Run a server from a class 
        """
        raise NotImplementedError

class GunicornWrapper(BaseApplication):
    def __init__(self, app, **kwargs):
        self.application = app
        self.options = kwargs
        super(GunicornWrapper, self).__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

class HypercornWrapper:
    def __init__(self, app, **kwargs):
        self.application = app
        self.worker_class = kwargs.pop('worker_class', 'asyncio')
        self.options = kwargs

        from hypercorn.config import Config
        self.config = Config().from_mapping(kwargs)

    def run(self):
        from hypercorn.asyncio import serve
        import asyncio
        
        if 'uvloop' in self.worker_class:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            loop.run_until_complete(serve(self.application, self.config))
        else:
            asyncio.run(serve(self.application, self.config))
