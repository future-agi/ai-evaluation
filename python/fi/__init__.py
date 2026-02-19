# FutureAGI AI Evaluation SDK
# This is a namespace package - extends path to include installed fi.* modules

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
