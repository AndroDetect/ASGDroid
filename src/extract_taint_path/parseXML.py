# coding:utf-8
import xml.sax
import os
from glob import glob
from tqdm import tqdm
import argparse
from os.path import basename


class TaintPathHandler( xml.sax.ContentHandler ):
	def __init__(self):
		self.CurrentData = ""
		self.pathList = list()
		self.fileName = ""
		self.writefolder = ""
	
	# Get the name of the file being parsed
	def setDocumentLocator(self, locator):
		self.fileName = os.path.basename(locator.getSystemId().replace(".xml", ""))

	def startDocument(self):
		#log.info("Process " + self.fileName + ".xml")
		pass
		
	# Element start event handler
	def startElement(self, tag, attrs):
		self.CurrentData = tag
		if tag == "TaintPath":
			self.pathList.append("PATH")
			
		if tag == "PathElement":
			if '<' in attrs["Statement"]:
				index1 = attrs["Statement"].find('<') + 1
				index2 = attrs["Statement"].rfind('>')
				self.pathList.append(attrs["Statement"][index1:index2])
			else:
				self.pathList.append(attrs["Statement"])
			
			
	# Element end event handler
	def endElement(self, tag):
		self.CurrentData = ""
		
	def endDocument(self):
		if not os.path.exists(self.fileName):
			file = os.path.join(self.writefolder, self.fileName + ".txt")
			#file = self.writefolder + self.fileName + ".txt"
			os.makedirs(self.writefolder, exist_ok=True)
			with open(file, "w") as f:
				for p in self.pathList:
					f.write(p + "\n")
		#log.info(self.fileName + " finished")
	
	# Directory for storing generated txt files
	def setWritePath(self, folder):
		self.writefolder = folder


def extractPath(xmlfile, writefolder):
	# Create an XMLReader
	parser = xml.sax.make_parser()
	# Turn off namespaces
	parser.setFeature(xml.sax.handler.feature_namespaces, 0)
	# Override ContextHandler
	handler = TaintPathHandler()
	# Set the write path for generated files
	handler.setWritePath(writefolder)
	parser.setContentHandler(handler)
	parser.parse(xmlfile)

		

if(__name__ == "__main__"):
	
	parser = argparse.ArgumentParser(description='Taint analysis for an given directory.')
	parser.add_argument("-tf", "--taint_file", help="A directory contains taint analysis xml files.", type=str, required=True) 

	args = parser.parse_args()

	xmls_dir = args.taint_file
	xmls_list = glob(f'{xmls_dir}/*.xml')

	dest_dir = xmls_dir.replace("taint_path_xml", "taint_path_txt")
	for xmlfile in tqdm(xmls_list):
		extractPath(xmlfile, dest_dir)
	
