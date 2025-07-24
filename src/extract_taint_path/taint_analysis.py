# coding:utf-8
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
import subprocess
import argparse
from os.path import basename


# Configure logging format
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s [%(levelname)s] %(message)s',
	handlers=[logging.FileHandler('../../Log/taint_anaylsis.log'), logging.StreamHandler()]
)

class APKProcessor:
	def __init__(self, input_dir):
		"""Initialize task queue"""
		self.task_queue = queue.Queue()
		self.input_dir = input_dir

		print(basename(self.input_dir))
		print(self.input_dir)

		_apk_dir = basename(self.input_dir)
		if len(_apk_dir) == 0:
			_apk_dir = self.input_dir.split("/")[-2]
		
		if self.input_dir.endswith('/'):
			self.output_dir = f'../../Raw/{self.input_dir.split("/")[-3]}/taint_path/{_apk_dir}/taint_path_xml'
		else:
			self.output_dir = f'../../Raw/{self.input_dir.split("/")[-2]}/taint_path/{_apk_dir}/taint_path_xml'
		print(self.output_dir)
		self.thread_num = 4
		logging.info(f"Initializing thread pool with {self.thread_num} workers")

	#def generate_tasks(self, root_dir="../../Dataset/", output_root="../../Data/flowdroid/output/"):
	def generate_tasks(self):
		"""Generate all pending tasks"""
		
		#apk_dir = os.path.join(self.input_dir, category)
		#output_dir = os.path.join(output_root, category)
		
		if not os.path.exists(self.input_dir):
			logging.warning(f"APK directory not exists: {self.input_dir}")
		
		# Create output directory
		os.makedirs(self.output_dir, exist_ok=True)
		
		# Iterate through APK files
		for apk_name in os.listdir(self.input_dir):
			if apk_name.endswith(".apk"):
				apk_path = os.path.join(self.input_dir, apk_name)
				task = (apk_path, self.output_dir)
				self.task_queue.put(task)
				logging.debug(f"Added task: {apk_path} -> {self.output_dir}")

		logging.info(f"Total tasks generated: {self.task_queue.qsize()}")

	def process_apk(self):
		"""Worker thread processing function"""
		while True:
			try:
				# Get task
				apk_path, output_dir = self.task_queue.get()
				apk_name = basename(apk_path)
				category = basename(os.path.dirname(apk_path))
				
				logging.info(f"Processing [{category}]: {apk_name}")
				
				# Build output path
				output_path = os.path.join(output_dir, apk_name.replace(".apk", ".xml"))

				# Construct command parameters
				cmd = f"java -jar ../../Lib/soot-infoflow-cmd-jar-with-dependencies.jar -a {apk_path} -s ../input/flowdroid/SourcesAndSinks.txt -p ../../Lib/platforms/ -ac /usr/local/java/jdk1.8.0_341/jre/lib/rt.jar -o {output_path} -pr PRECISE -ct 180 -dt 180 -rt 180 " #  -ct 60 -dt 60 -rt 60 
				
				# Execute command and capture output
				try:
					result = subprocess.run(
						cmd,
						
						stdout=subprocess.PIPE,
						shell=True,
						timeout=300  # 5 minutes timeout
					)
					logging.debug(f"Processed [{category}] {apk_name} successfully\nOutput: {result.stdout.decode()}")
				except subprocess.CalledProcessError as e:
					logging.error(f"Failed to process [{category}] {apk_name}\nError: {e.output.decode()}")
				except subprocess.TimeoutExpired:
					logging.error(f"Timeout processing [{category}] {apk_name}")

				self.task_queue.task_done()

			except Exception as e:
				logging.error(f"Unexpected error: {str(e)}")
				self.task_queue.task_done()
				break

	def run(self):
		"""Start processing workflow"""
		self.generate_tasks()
		
		# Create daemon threads, automatically end when main thread exits
		threads = []
		for i in range(self.thread_num):
			t = threading.Thread(target=self.process_apk, daemon=True)
			t.start()
			threads.append(t)
		
		# Wait for all tasks to complete
		self.task_queue.join()
		logging.info("All tasks completed")

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Taint analysis for an given directory.')
	parser.add_argument("-d", "--dir", help="A directory with APK(s) to analyze.", type=str, required=True) 

	args = parser.parse_args()

	processor = APKProcessor(args.dir)
	processor.run()