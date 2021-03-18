import sys
from datetime import datetime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKPURPLE = '\033[0;35m'
    OKADVISORY = '\033[1;36m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def success(msgstring,destination=sys.stdout,verbose=True):
	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
	if type(destination) is not list:
		destination=[destination]
	if verbose==False:
		return
	for dest in destination:
		print(bcolors.OKGREEN+bcolors.BOLD+"[%s] | Success | "%now+ bcolors.ENDC+msgstring,file=dest)

def status(msgstring,destination=sys.stdout,verbose=True,end=None,require_input=False):

	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

	if type(destination) is not list:
		destination=[destination]

	if verbose==False:
		return

	input_from_user=None

	for dest in destination:
		if end is None:
			if require_input:
				if dest==sys.stdout:
					inp_f=raw_input if sys.version_info[0]<3 else input
					input_from_user=inp_f(bcolors.OKBLUE +bcolors.BOLD+"[%s] | Input   | "%now+bcolors.ENDC + msgstring)
				else:
					print (bcolors.OKBLUE +bcolors.BOLD+"[%s] | Status  | "%now+bcolors.ENDC + msgstring,file=dest)
					
			else:
				print (bcolors.OKBLUE +bcolors.BOLD+"[%s] | Status  | "%now+bcolors.ENDC + msgstring,file=dest)
		else:
			print (bcolors.OKBLUE +bcolors.BOLD+"[%s] | Status  | "%now+bcolors.ENDC + msgstring,file=dest,end="\r")

	if input_from_user!=None:
		return input_from_user

def advisory(msgstring,destination=sys.stdout,verbose=True):
	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
	if type(destination) is not list:
		destination=[destination]
	if verbose==False:
		return
	for dest in destination:
		print (bcolors.OKADVISORY +bcolors.BOLD+"[%s] | Advise  | "%now+bcolors.ENDC + msgstring,file=dest)

def error(msgstring,error=False,errorType=RuntimeError,destination=sys.stdout,verbose=True):
	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
	if type(destination) is not list:
		destination=[destination]
	if verbose==False:
		return
	if error==False:
		for dest in destination:
			print (bcolors.WARNING +bcolors.BOLD+"[%s] | Warning | "%now+bcolors.ENDC + msgstring,file=dest)
	else:
		raise errorType(msgstring)

def progress_spinner(message,progress,speed=1./5000):
	status ("%s%s"%(message,'/-\|'[int(progress*speed)%4]),end="\r")

