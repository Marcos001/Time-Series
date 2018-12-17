# append directory
import sys, os

_path = os.getcwd()+'/'
print(' path : ', _path)
sys.path.append(_path)
print(' diretórios visiveis ao interpretador : ', sys.path)
