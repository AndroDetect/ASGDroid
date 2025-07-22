import glob
import os 
from subprocess import Popen, PIPE
import parseGraph
import shlex
import argparse
from os.path import basename

def parseargs():
	parser = argparse.ArgumentParser(description = "Minimum RAM requirement is 4G.")
	parser.add_argument("-f", "--file", help="A directory with APK(s) to analyze.", type=str, required=True) 
	parser.add_argument("-d", "--dir", help="The path to your Android platform directory", type=str, required=True)
	args = parser.parse_args()
	return args


def _make_dirs(_base_dir):
	try:
		os.makedirs(_base_dir + "/graphs")

	except OSError:
		print ("One or more of the default directory already exists. Skipping directory creation...")


def _repeated_function(app, _app_dir):
	try:
		if os.path.isfile(app + ".txt"):

			_graphFile = parseGraph.parse_graph(app + ".txt", _app_dir)
			os.remove(app + ".txt")
		else:
			print ("There was an error extracting call graphs from", app)
	except Exception as err:
		print (err)


def main():
	#_base_dir = os.getcwd()
	_base_dir = "../../Raw"
	
	apps = parseargs()

	if os.path.isdir(apps.file):
		_apk_dir = apps.file.split("/")[-1]
		if len(_apk_dir) == 0:
			_apk_dir = apps.file.split("/")[-2]

		print('='*50)
		print(apps.file)
		

		if apps.file.endswith('/'):
			_app_dir = _base_dir + "/" + apps.file.split("/")[-3] + "/" + "call_graph" + "/" + _apk_dir
		else:
			print('*'*50)
			print(basename(apps.file))
			_app_dir = _base_dir + "/" + apps.file.split("/")[-2] + "/" + "call_graph" + "/" + _apk_dir
		_make_dirs(_app_dir)

		for app in glob.glob(apps.file + "/*.apk"):
			cmd = "java -Xms4g -Xmx16g -XX:+UseConcMarkSweepGC Appgraph " + app + " " + apps.dir
			ran = Popen(shlex.split(cmd))
			while 1:
				check = Popen.poll(ran) 
				if check is not None:		#check if process is still running
					break
			if ran.communicate()[1]:
				_repeated_function(app, _app_dir)
			else:
				_repeated_function(app, _app_dir)
	else:
		print('Please input a valid directory.')


if __name__ == "__main__":
	main()