import sys
import importlib

def check_packages(packages):
	for package in packages:
		try:
			module = importlib.import_module(package)
			print('{:20s}:\t{}'.format(package, module.__version__))
		except:
			print("{} has to be installed".format(package))