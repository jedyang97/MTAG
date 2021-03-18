from itertools import product
from standard_grid import log
from collections import OrderedDict 
import hashlib
import os
import pickle
import sys
import time
import math
import random


def get_hash(in_str):

	hash_object = hashlib.sha512(in_str.encode("utf-8"))
	return str(hash_object.hexdigest())

#TODO: Ensure grid is not modified if already generated
class Grid:

	def __init__(self,entry_script_, grid_root_,resume_on_retry_=True,grid_operator_=product,description_="Grid search object"):
		if os.path.isfile(entry_script_) is False:
			log.error("Entry point does not exist. Exiting ...!",error=True)
		self.entry_script=entry_script_
		self.entry=os.path.relpath(self.entry_script,grid_root_)

		self.grid_root=os.path.relpath(grid_root_,os.sep.join(sys.argv[0].split(os.sep)[:-1]))
		if os.path.isdir(self.grid_root) is False:
			os.makedirs(self.grid_root) 

		self.resume_on_retry=resume_on_retry_
		self.grid_operator=grid_operator_
		self.description=description_

		self.grid_parameters=OrderedDict()
		self.shell_instances_generated=False
		self.grid_generated=False
		self.grid_saved=False

	def register(self,key,value):
		if self.grid_generated==True:
			log.error("Grid already generated, cannot register any more hypers. Exiting ...!",error=True)
		if type(key)!=str:
			log.error("Key to be registered is not a string. Exiting ...!",error=True)
		if type(value)!=list:
			log.error("Value to be registered is not a list. Exiting ...!",error=True)
		
		self.grid_parameters[key]=value
		log.status("%s registered in grid."%key)

	def generate_grid(self):
		grid_list=[]
		try:
			for key in list(self.grid_parameters.keys()):
				grid_list.append(self.grid_parameters[key])
		except:
			log.error("Grid cannot be generated, check your parameters. Exiting ...!",error=True)

		self.grid=list(self.grid_operator(*grid_list))
		self.grid_generated=True


		def create_grid_hash__():
			total_commands_str=[]
			#TODO: This can be done better - just need the list itself and not combinations
			for i in range(len(self.grid)):
				grid_args=self.gen_args(i)
				final_command=grid_args
				total_commands_str.append(final_command)
				self.grid_hash=get_hash("".join(total_commands_str))

		create_grid_hash__()

		self.grid_dir=os.path.join(self.grid_root,self.grid_hash)
		if os.path.isdir(self.grid_dir) is True:
			log.error("Identical grid already exists. Exiting ...!",error=True)

		log.success("Grid successfully generated.")

	def shuffle_grid(self):
		if self.grid_generated==False:
			log.error("Grid was not generated, cannot shuffle. Exiting ...!",error=True)

		random.shuffle(self.grid)

	def __getitem__(self,i):
		if self.grid_generated==False:
			log.error("Grid is not generated, please call .generate first. Exiting ...!",error=True)
		else:
			return self.grid_item(i) 

	def grid_item(self,i):
		if self.grid_generated==False:
			log.error("Grid is not generated, cannot get item. Exiting ...!",error=True)
		output={}
		for key,j in zip(list(self.grid_parameters.keys()),range(len(list(self.grid_parameters.keys())))):
			output[key]=self.grid[i][j]
		return output 

	def gen_args(self,i):
		args=""
		item=self[i]
		for key in list(item.keys()):
			args+=" --%s %s"%(str(key),str(item[key]))
		return args


	def generate_shell_instances(self,prefix="",postfix=""):
		if self.shell_instances_generated:
			log.error("Shell instances already generated. Exiting ...!")

		os.makedirs(self.grid_dir)
		instances_dir=os.path.join(self.grid_dir,"instances/")

		entry_point_relative_to_instance=os.path.join("../../../",self.entry)

		def write_shell_instance_content__(fhandle,command,command_hex):
			fhandle.write ("#!/bin/sh\n")
			fhandle.write ("run_grid_instance (){\n")
			fhandle.write ("echo \"STARTED\" > shell_instance_started_signal\n")
			fhandle.write("\t"+command+" \n ")
			fhandle.write("\t%s > %s\n"%("echo $?","STANDARDGRID_instance_output"))
			fhandle.write ("rm shell_instance_started_signal\n")
			fhandle.write ("}\n")
			fhandle.write ("run_grid_instance")

		def write_instance_parameters__(fhandle,grid_instance):
			pickle.dump(grid_instance,fhandle)

		for i in range(len(self.grid)):
			grid_instance=self.gen_args(i)
			command=prefix+" "+entry_point_relative_to_instance+" "+grid_instance
			command_hex=get_hash(command)
			command=command+" "+postfix
			command_dir=os.path.join(instances_dir,command_hex)
			os.makedirs(command_dir)

			local_sh_name=os.path.join(command_dir,command_hex+".sh")
			write_shell_instance_content__(open(local_sh_name,"w"),command,command_hex)

			instance_pkl_fname=os.path.join(command_dir,command_hex+".pkl")
			write_instance_parameters__(open(instance_pkl_fname,"wb"),self[i])

		
		self.shell_instances_generated=True
		log.success("Shell instances created for grid in %s/instances"%self.grid_dir)

	def get_status(self,success_codes=[0]):

		started=[]
		finished=[]
		failed=[]
		not_started=[]

		command_hexes = [gi for gi in os.listdir(os.path.join(self.grid_dir,"instances/")) if os.path.isdir(os.path.join(self.grid_dir,"instances/",gi))]

		for command_hex in command_hexes:
			command_dir=os.path.join(self.grid_dir,"instances/",command_hex+"/")
			started_signal_fname=os.path.join(command_dir,"shell_instance_started_signal")
			output_code_fname=os.path.join(command_dir,"STANDARDGRID_instance_output")
			try:
				output_code=int(open(output_code_fname,"r").read())
			except:
				if os.path.exists(started_signal_fname): 
					started.append(command_hex)
				else:
					not_started.append(command_hex)
				continue
			if output_code in success_codes:
				finished.append(command_hex)
			else:
				failed.append(command_hex)

		log.status("Not started:	%.2f%%"%(float(len(not_started))*100/len(command_hexes)))
		log.status("Unfinished*:	%.2f%%"%(float(len(started))    *100/len(command_hexes)))
		log.status("Finished:		%.2f%%"%(float(len(finished))   *100/len(command_hexes)))
		log.status("Failed:		%.2f%%"%(float(len(failed))     *100/len(command_hexes)))

		return started,finished,failed,not_started

	def resume_as_before(self,hard_resume=False):
		self.create_runner(**{**self.last_run_params,"hard_resume":hard_resume})

	def resume(self,fraction=1.0,num_runners=1,runners_prefix=["sh"],parallel=1,hard_resume=False):
		self.create_runner(fraction=fraction,num_runners=num_runners,runners_prefix=runners_prefix,parallel=parallel,hard_resume=hard_resume)

	def __nullify_previous_instance_runs(self,nullification_list,strict=True):
		for command_hex in nullification_list:
			shell_instance_dir=os.path.join(self.grid_dir,"instances/",command_hex+"/")
			if strict:
				removable_content=os.listdir(shell_instance_dir)
				removable_content.remove(os.path.join(command_hex+".sh"))
				removable_content.remove(os.path.join(command_hex+".pkl"))
			else:
				removable_content=[os.path.join("STANDARDGRID_instance_output")]
			for r in removable_content:
				if os.path.exists(r):
					os.remove(os.path.join(output_code_fname,r))
				else:
					pass

	def create_runner(self,fraction=1.0,num_runners=1,runners_prefix=["sh"],parallel=1,hard_resume=False):

		if parallel>3:
			log.error("Parallel cannot be higher than 3.",error=True)

		if fraction>1 or fraction<0:
			log.error("Fraction not in range [0,1].",error=True)

		if num_runners==None:
			num_runners=len(self.grid)

		if len(runners_prefix)==1:
			runners_prefix=runners_prefix*num_runners

		if len(runners_prefix)!=num_runners:
			log.error("Mismatch between num_runners and runners_prefix arguments. Exiting ...!",error=True)

		last_run_params={"fraction":fraction,"num_runners":num_runners,"runners_prefix":runners_prefix,"parallel":parallel}
		self.last_run_params=last_run_params

		if hard_resume:
			while True:
				log.status("Specified hard_resume, are you sure you want to add the unfinished instances to the grid? (y,n)")
				permission = input('Enter y|n:')
				if permission=="y":
					break
				elif permission=="n":
					exit()
				else:
					pass

		started,finished,failed,not_started=self.get_status()

		if hard_resume:
			command_hexes=started+failed+not_started
		else:
			command_hexes=failed+not_started

		if len(command_hexes)==0:
			log.advisory("No more instances remaining. Exiting ...!")
			exit()

		#Ensure no script output is recorded, therefore all the instances will be seen as not-started. 
		self.__nullify_previous_instance_runs(command_hexes)

		#If the central command directory does not exist, then recreate it
		central_command_dir=os.path.join(self.grid_dir,"central/")
		if not os.path.exists(central_command_dir):
			os.makedirs(central_command_dir)

		temp_counter=0
		while (True):
			attempt="attempt_%d/"%temp_counter
			attempt_dir=os.path.join(self.grid_dir,"central/",attempt)
			if os.path.exists(attempt_dir):
				temp_counter+=1
			else:
				break

		os.makedirs(attempt_dir)

		group_dir=os.path.join(self.grid_dir,"central/",attempt,"groups/")
		if not os.path.exists(group_dir):
			os.makedirs(group_dir)

		main_handle=open(os.path.join(attempt_dir,"main.sh"),"w")

		def write_main_entries__(main_handle,entry):
			main_handle.write("cd ./groups/\n")
			main_handle.write("%s\n"%entry)
			main_handle.write("cd - > /dev/null\n")

		split_len=math.ceil((len(self.grid)*fraction)/num_runners)
		run_counter=0

		for i in range(num_runners):
			this_group="group_%d.sh"%i
			this_hexes=command_hexes[i*split_len:(i+1)*split_len]
			this_group_fname=os.path.join(group_dir,this_group)
			group_handle=open(this_group_fname,"w")
			for this_hex in this_hexes: 
				group_handle.write("cd ../../../instances/%s/ && echo Running %s && %s %s.sh > %s.stdout && cd - > /dev/null\n"%(this_hex,this_hex,runners_prefix[i],this_hex,this_hex))
			write_main_entries__(main_handle,"cat %s | xargs -L 1 -I CMD -P %d bash -c CMD &"%(this_group,parallel))
		main_handle.write("wait")
		main_handle.close()

		log.success("Grid runners established under %s"%os.path.join(self.grid_dir,attempt))

	def apply(self,apply_fn,input_file,output_file):
		started,finished,failed,not_started=self.get_status()

		if len(finished)==0:
			log.error("No results to compile yet. Exiting ...!",error=True)

		for command_hex in finished:
			in_fpath=os.path.join(self.grid_dir,"instances/",command_hex+"/",input_file)
			out_fpath=os.path.join(self.grid_dir,"instances/",command_hex+"/",output_file)
			if not os.path.isfile(in_fpath):
				log.warning("%s does not have the %s file."%(command_hex,input_file))
				continue

			try:
				apply_fn(in_fpath,out_fpath)
			except Exception as e:
				log.error("Cannot apply the given function to instance %s with error %s ...!"%(command_hex,str(e)),error=False)

		log.success("Function application finished.")
		


	def json_interpret(self,main_res_file,output_file):
		started,finished,failed,not_started=self.get_status()

		if len(finished)==0:
			log.error("No results to compile yet. Exiting ...!",error=True)

		import json
		res_list=[]
		for command_hex in finished:
			main_res_fpath=os.path.join(self.grid_dir,"instances/",command_hex+"/",main_res_file)
			if not os.path.isfile(main_res_fpath):
				log.warning("%s does not have the %s file."%(command_hex,main_res_file))
				continue
			main_res_content=open(main_res_fpath,"r").read()
			main_res=json.loads(main_res_content)

			in_params_fpath=os.path.join(self.grid_dir,"instances/",command_hex+"/",command_hex+".pkl")
			in_params=pickle.load(open(in_params_fpath,"rb"))
			for key in in_params:
				main_res["STDGRID_%s"%key]=in_params[key]

			res_list.append(main_res)
			res_list[-1]["STDGRID_command_hex"]=command_hex
			res_list[-1]["STDGRID_grid_hex"]=os.path.basename(self.grid_dir)

		self.__write_res_to_csv(res_list,output_file)

	def __write_res_to_csv(self,res_list,output_fname):
		import csv

		with open(output_fname,"w") as output:

			csv_writer=csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			#Writing the headers first
			res0=res_list[0]
			header=[]
			for key in res0:
				header.append(key)
			csv_writer.writerow(header)
	
			for res in res_list:
				row=[]
				for key in res0:
					row.append(res[key])
				csv_writer.writerow(row)

		log.success("Results gathered in %s"%output_fname)

	def save(self,dump_fname):
		import pickle
		self.grid_saved=True
		pickle.dump(self,open(dump_fname,"wb"))

	def __del__(self):
		if self.grid_saved or not self.grid_generated:
			return
		else:
			if os.path.isfile(".%s.pkl"%self.grid_hash):
				return
			self.save(".%s.pkl"%self.grid_hash)
