import argparse
from standard_grid import log

class ArgParser:
	""" The class for building the parameter space of a machine learning code 
	# Properties 
		parameters

	# Methods 
		register_parameter
		compile_argparse
		remove_parameter

	# Class Methods
		
	# Raises
		RuntimeError: if a parameter is already registered
	"""

	def __init__(self):
		self.parameters={}
		
	
	def register_parameter(self,__name,__type,__default,__help=""):
		""" Registers a parameter for machine learning code 

		# Arguments
			__name: The name of the variable to be registered.
			__type: The type of the variable. 
			__default: Default value to be passed to the variable. 
			__help: Helper for the parameter.
 
		#Returns
			

		"""
		if __name in list(self.parameters.keys()):
			log.error("Parameter already registered. Exiting ...!",error=True)
		
		self.parameters[__name]=[__type,__default,__help]
		
		
	def compile_argparse(self,dict_style=False):
		""" Compiles the parameters using argparse method 

		# Arguments

		#Returns
			An argparse object.

		"""

		if len(list(self.parameters.keys()))==0:
			log.error("No parameters registered. Exiting ...!",error=True)

		parser = argparse.ArgumentParser()
		for key in list(self.parameters.keys()):
			parser.add_argument(str(key), type=self.parameters[key][0], default=self.parameters[key][1],help=self.parameters[key][2])

		if dict_style is False:
			return parser.parse_args()
		else:
			return vars(parser.parse_args())

	def remove_parameter(key):
		""" Removes parameter from the ArgParser object 

		# Arguments
			key: The key to be removed.

		#Returns

		"""

		del self.parameters[key]
