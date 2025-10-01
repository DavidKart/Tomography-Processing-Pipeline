import os
import glob
import re
import numpy as np
import subprocess

startingDirectory = os.path.realpath(os.path.dirname(__file__))
os.chdir(startingDirectory)

from scripts import trainForAllTomos
from scripts import predictForAllTomos


#_______________________initialize input parameters__________________
reconstructHalfMaps = True #True #Creation of tomograms and tomogram half maps from raw data
trainCryoCARE = True #False #cryoCARE train
predictCryoCARE = True #cryoCRAE predict

# general
apix = "read" #read from raw .tif files if set to "read". If that information is incorrect, just input give number.
mdocPreprocessing = "mdocs/*.mdoc" #"*.mdoc" takes only data in the present directory. Otherwise just give one mdoc or specifiy multiple, for example "TS[2-4].mdoc"
mdocryoCARETrain = "mdocsTrain/*.mdoc" #choose a small subset of representative tomos
mdocryoCAREPredict = "mdocs/*.mdoc"
ending="tiff" #tif, tiff, mrc, eer
gain="" #input name of gain (in .mrc format! if ".dm4" convert with: 'dm2mrc input_ref.dm4 output_ref.mrc'

#MotionCor3
flipy = 2 #2 corresponds to y-flip. Check 'MotionCor3 --help' for more information. Use 0 for no flip.
mcBinning = 2 #binning in MotionCor3. Probably makes things a bit faster. 1 means no binning!
gpus = "-Gpu 0 1" #MotionCore3. For AreTomo there does not seem to be an option yet for multiple gpu usage.

#areTomo2
binning = 4 #areTomo2 binning. Overall binning should be 'mcBinning*binning'.
locAlign = "-Patch 4 5"
applyTiltAxisCorrection = 1



#expert options__________
divByTilt = False

#executables
motionCor = "MotionCor3"
areTomo = "AreTomo2"
imodNewStack = "newstack"

#CTF correction - correct CTF for aligned stacks with ctfplotter!
include_CTFcorrection = False
expectedDefocus = "read" #read automatically from mdoc file
TiltAxisRotation = "read" #read automatically from mdoc file

#only applicable if ending == "eer"
eerSampling = "1"
fmIntFile = "fmintFile"

#________________________run_________________________________________
#cryoCARE #####
cryoCAREpath = os.path.realpath(os.path.join(startingDirectory, "cryoCARE")) #dir where to run cryoCARE. Just leave it.
if divByTilt and (len(locAlign) > 1):
	import mrcfile
	locAlign = " "
	print("align local patches only when averaging frames")


if applyTiltAxisCorrection == 1:
    applyTiltAxisCorrection = " -TiltCor 1"
else:
    applyTiltAxisCorrection = " -TiltCor 0"
    
def splitAlnFile(alnFileName, evenalnFileName, oddalnFileName):
	print("apply split of aln file")
	with open(alnFileName, "r") as readfile:
		writefileEven = open(os.path.join(evenalnFileName), "w")
		writefileOdd =  open(os.path.join(oddalnFileName), "w")
		count = 0
		for index_i, i in enumerate(readfile):
			if i[0] == "#":
				if "RawSize" in i:
					match = re.findall(r'\d+', i)[2]
					writefileEven.write(i.replace(match, str(int(np.ceil(int(match)/2)))))
					writefileOdd.write(i.replace(match, str(int(np.floor(int(match)/2)))))
					continue
				writefileEven.write(i)
				writefileOdd.write(i)
				continue
			match = re.search(r'\d+', i)
			match = match.group()
			if int(count)%2 == 0:
				# writefileEven.write(i)
				writefileEven.write(i.replace(match, str(int(np.ceil(int(match)/2)))))
			else:
				# writefileOdd.write(i)
				writefileOdd.write(i.replace(match, str(int(np.floor(int(match)/2)))))
			count += 1

		writefileEven.close()
		writefileOdd.close()


def split_stackedFile(inFile, evenOut, oddOut):
	mrcfiledata = mrcfile.open(inFile, permissive=True)
	# apix = float((mrcfiledata.voxel_size).x) # 1.52#1.9#2.394#1.52#2.394#1.52#1.9#1.52#
	data = mrcfiledata.data
 
	data1 = data[::2, :, :]
	data2 = data[1::2, :, :]

	file1 = mrcfile.new(evenOut, overwrite=True)
	file2 = mrcfile.new(oddOut, overwrite=True)

	mapToSave = np.float32(data1)
	file1.set_data(mapToSave)
	# file1.voxel_size = apix
	file1.close()

	mapToSave = np.float32(data2)
	file2.set_data(mapToSave)
	# file2.voxel_size = apix
	file2.close()


def runMotionCorr(fileList):
	averagedList=[]
	
	if (mcBinning != 1) and (mcBinning != 0):
		mcBinningIn = " -FtBin " + str(mcBinning) + " "
	else:
		mcBinningIn = ""

	eerSamplingIn = ""
	fmIntFileIn = ""
	if ending == "eer":
		eerSamplingIn = " -EerSampling " + eerSampling
		fmIntFileIn = " -FmIntFile " + fmIntFile
		inFormat = " -InEer "
	else:
		if "tif" in ending:
			inFormat = " -InTiff "
	
		else:
			inFormat = " -InMrc "
  
	flipIn = ""
	if flipy:
		flipIn = " -FlipGain " + str(flipy) + " "


	
	inGain = ""
	if len(gain) != 0:
		inGain = " -Gain " + str(gain) + " "
			
	
	splitSum = " -SplitSum 1 "
	if divByTilt:
		splitSum = " "
   

	for i in fileList:
		if i == gain: continue
		if "averaged" in i: continue
		outfile = i[:-4]+"_averaged.mrc"
		averagedList.append(outfile)
		os.system(motionCor+" " + inFormat + i + " -outMrc " + outfile + inGain + flipIn + fmIntFileIn + eerSamplingIn + mcBinningIn + "-Iter 7 -Tol 0.5" + splitSum + gpus + " -LogFile " + i + ".log")
	
	return averagedList

def rec(apix, expectedDefocus, TiltAxisRotation):
	#1. read mdoc files
	mdocFiles = glob.glob(os.path.join(mdocPreprocessing))
	if len(mdocFiles) == 0:
		print("NO MDOCS FOUND. EXIT.")
		return

	for files in mdocFiles:
		os.chdir(startingDirectory)
		tomoName = "out_" + files.split("/")[-1].split(".")[0]
		
	 
		angles=[]
		names=[]
		current_tiltAxisAngle = 0
		current_expectedDefocus = 0
		with open(files, "r") as readfile:
			for line in readfile:
				if include_CTFcorrection:
					matchTAA = re.search(r"Tilt axis angle\s*=\s*([0-9.]+)", line)
					matchTDE = re.search(r"TargetDefocus\s*=\s*(-?\d+)", line)
					if matchTAA:
						current_tiltAxisAngle = float(matchTAA.group(1))
					if matchTDE:
						current_expectedDefocus = float(matchTDE.group(1)) * -1000
						
				if "TiltAngle" in line:
					angle=line.strip()
					angle= re.findall(r"[-+]?\d*\.\d+|\d+", angle)
					angles.append(float(angle[0]))
				if "SubFramePath" in line:
					fileName = line.strip()
					fileName = fileName.split("\\")[-1]
					names.append(fileName)


		# fill variables correctly
		if apix == "read":
			cmd = f"header {names[0]} | grep 'Pixel spacing' | awk '{{print $4}}'"
			apix = subprocess.check_output(cmd, shell=True, text=True).strip()
		apix = float(apix)
   
		if include_CTFcorrection:
			if expectedDefocus != "read":
				current_expectedDefocus = expectedDefocus
			if TiltAxisRotation != "read":
				current_tiltAxisAngle = TiltAxisRotation
			#check if all is fine
			if isinstance(apix, str) or isinstance(current_expectedDefocus, str) or isinstance(current_tiltAxisAngle, str):
				print("ERROR for given apix, tilt axis angle or expected defocus. Cannot apply CTF correction. skip tilt series.")



		#2. output dirs
		#make output directory
		oddPrefix = tomoName+"_odd"
		evenPrefix = tomoName+"_even"
		if not os.path.exists(evenPrefix):
			os.makedirs(evenPrefix)
		if not os.path.exists(oddPrefix):
			os.makedirs(oddPrefix)

		oddPath = os.path.join(startingDirectory, oddPrefix)
		evenPath = os.path.join(startingDirectory, evenPrefix)

		print("made dirs")
   		
     
		#3. produce full MotionCorrected Tomogram!
		runMotionCorr(names)

		#4. get .rawtlt file
		rawtlt=tomoName + ".rawtlt"
		indices = sorted(range(len(angles)), key=lambda k: angles[k])
		angles = [angles[j] for j in indices] #sorted(angles)
		stackList = ""
		stackListSorted = [names[j][:-4]+"_averaged.mrc" for j in indices]
		for check in stackListSorted:
			if not os.path.exists(check):
				index = stackListSorted.index(check)
				stackListSorted.pop(index)
				angles.pop(index)


		for index_j, j in enumerate(stackListSorted):
			stackList += j
			if index_j != (len(stackListSorted)-1):
				stackList += " "


		with open(rawtlt, "w") as writeFile:
			for j in angles:
				writeFile.write(str(j)+"\n")


		#prepare
		if not divByTilt:
			os.system("mv *EVN* " + evenPath)
			os.system("mv *ODD* " + oddPath)
		os.system("cp " + rawtlt + " " + evenPath)
		os.system("cp " + rawtlt + " " + oddPath)

		#5. run newStack
		stackOutFileName = tomoName + "_stacked.mrc"
		os.system(imodNewStack+ " " + stackList + " " + stackOutFileName)

		if include_CTFcorrection:
			print("PERFOM CTF ESTIMATION OF TILT SERIES")
			print("pixel size: " + str(apix) + "\ntilt axis angle: "+ str(current_tiltAxisAngle)+ "\nexpected defocus: " + str(current_expectedDefocus))
			outDefFile = "defocusFile.txt"
			ctf_stackOutFileName = "ctfCorrected_" + stackOutFileName
			
			os.system("ctfplotter -inp " + stackOutFileName + " -vo 300 -cs 2.7 -pi " + str(apix) + " -an "+ rawtlt + " -exp " + str(current_expectedDefocus) + " -defF " + str(outDefFile) + " -aA " + str(current_tiltAxisAngle)+ " -au 1,1 -use -ba 2 -va -fit 4 -sa -sA")
			print("PERFOM CTF CORRECTION OF TILT SERIES")
			os.system("ctfphaseflip -inp " + stackOutFileName + " -o " + ctf_stackOutFileName +" -an "+ rawtlt + " -defF " + str(outDefFile) + " -gpu 0 -defT 1 -iW 1 -pi " + str(apix) + " -vo 300 -cs 2.7 -am 0.07 -ax " + str(current_tiltAxisAngle))
			stackOutFileName = ctf_stackOutFileName

			os.system("cp " + outDefFile + " " + evenPath)
			os.system("cp " + outDefFile + " " + oddPath)
  
		#6. run AreTomo for full tomogram
		apixAreTomo = ""
		if not isinstance(apix, str):
			apixAreTomo = " -PixSize " + str(mcBinning*apix)
		bin = "-OutBin " + str(binning) 
		alignedName=tomoName+"_rec_tomo.mrc"
		os.system(areTomo+" -InMrc "+ stackOutFileName + " -OutMrc " + alignedName + apixAreTomo + " -VolZ 1200 " + bin + " -AngFile " + rawtlt + applyTiltAxisCorrection + " -DarkTol 0.5 -OutXf 1 -Wbp 1 -FlipVol 1 " + locAlign + " >" + tomoName + "_logFile.txt")
		ctf_stackOutFileNameFull = stackOutFileName
		#cp result
		alnFileName = stackOutFileName[:-4] + ".aln"
  
		stackedFileEVN_name = "even_" + stackOutFileName
		stackedFileNameODD_name = "odd_" + stackOutFileName
		alnFileNameEVN = alnFileName[:-4] + "EVN.aln"
		alnFileNameODD = alnFileName[:-4] + "ODD.aln"
   
		if not divByTilt:
			os.system("cp " + alnFileName + " " + evenPath)
			os.system("cp " + alnFileName + " " + oddPath)
			alnFileNameEVN, alnFileNameODD = alnFileName, alnFileName
		else:
			alnFileNameEVN = os.path.join(evenPath, alnFileNameEVN)
			alnFileNameODD = os.path.join(oddPath, alnFileNameODD)
			splitAlnFile(alnFileName, alnFileNameEVN, alnFileNameODD)
			split_stackedFile(stackOutFileName, os.path.join(evenPath, stackedFileEVN_name), os.path.join(oddPath, stackedFileNameODD_name))

   		#7. running even/odd reconstructions
		os.chdir(evenPath)
		if not divByTilt:
			stackList_even = stackList.replace("averaged", "averaged_EVN")
			stackOutFileName = "even_" + tomoName + "_stacked.mrc"
			os.system(imodNewStack + " " + stackList_even + " " + stackOutFileName)
		
			if include_CTFcorrection:
				print("PERFOM CTF CORRECTION OF EVEN TILT SERIES")
				# print("pixel size: " + str(apix) + "\ntilt axis angle: "+ str(current_tiltAxisAngle)+ "\nexpcected defocus: " + str(current_expectedDefocus))
				# outDefFile = "defocusFile.txt"
				ctf_stackOutFileName = "ctfCorrected_" + stackOutFileName
				
				# os.system("ctfplotter -inp " + stackOutFileName + " -vo 300 -cs 2.7 -pi " + str(apix) + " -an "+ rawtlt + " -exp " + str(current_expectedDefocus) + " -defF " + str(outDefFile) + " -aA " + str(current_tiltAxisAngle)+ " -au 1,1 -use -ba 2 -va -fit 4 -sa -sA")
				os.system("ctfphaseflip -inp " + stackOutFileName + " -o " + ctf_stackOutFileName +" -an "+ rawtlt + " -defF " + str(outDefFile) + " -gpu 0 -defT 1 -iW 1 -pi " + str(apix) + " -vo 300 -cs 2.7 -am 0.07 -ax " + str(current_tiltAxisAngle))
				stackedFileEVN_name = ctf_stackOutFileName

  
		# os.system(areTomo+" -InMrc "+ stackOutFileName + " -OutMrc even_" + alignedName + " -VolZ 1200 " + bin + "  -AngFile " + rawtlt + " -AlnFile " + alnFileName +" -TiltCor 1 -DarkTol 0.01 -OutXf 1 -Wbp 1 -FlipVol 1" + locAlign + ">" + tomoName + "_even_logFile.txt")
		os.system(areTomo+" -InMrc "+ stackedFileEVN_name + " -OutMrc even_" + alignedName + apixAreTomo + " -VolZ 1200 " + bin + " -AlnFile " + alnFileNameEVN + applyTiltAxisCorrection + " -DarkTol 0.5 -OutXf 1 -Wbp 1 -FlipVol 1 " + locAlign + " >" + tomoName + "_even_logFile.txt")


		os.chdir(oddPath)
		if not divByTilt:
			stackList_odd = stackList.replace("averaged", "averaged_ODD")
			stackOutFileName = "odd_" + tomoName + "_stacked.mrc"
			os.system(imodNewStack + " " + stackList_odd + " " + stackOutFileName)
   
			if include_CTFcorrection:
				print("PERFOM CTF CORRECTION OF ODD TILT SERIES")
				# print("pixel size: " + str(apix) + "\ntilt axis angle: "+ str(current_tiltAxisAngle)+ "\nexpcected defocus: " + str(current_expectedDefocus))
				# outDefFile = "defocusFile.txt"
				ctf_stackOutFileName = "ctfCorrected_" + stackOutFileName
				
				# os.system("ctfplotter -inp " + stackOutFileName + " -vo 300 -cs 2.7 -pi " + str(apix) + " -an "+ rawtlt + " -exp " + str(current_expectedDefocus) + " -defF " + str(outDefFile) + " -aA " + str(current_tiltAxisAngle)+ " -au 1,1 -use -ba 2 -va -fit 4 -sa -sA")
				os.system("ctfphaseflip -inp " + stackOutFileName + " -o " + ctf_stackOutFileName +" -an "+ rawtlt + " -defF " + str(outDefFile) + " -gpu 0 -defT 1 -iW 1 -pi " + str(apix) + " -vo 300 -cs 2.7 -am 0.07 -ax " + str(current_tiltAxisAngle))
				stackedFileNameODD_name = ctf_stackOutFileName
    
		os.system(areTomo+" -InMrc "+ stackedFileNameODD_name + " -OutMrc odd_" + alignedName + apixAreTomo + " -VolZ 1200 " + bin + " -AlnFile " + alnFileNameODD + applyTiltAxisCorrection + " -DarkTol 0.5 -OutXf 1 -Wbp 1 -FlipVol 1 " + locAlign + " >" + tomoName + "_odd_logFile.txt")

		#create rawTomos dir and move half maps in preparation for cryoCARE
		if not os.path.exists(cryoCAREpath) or not os.path.isdir(cryoCAREpath):
			os.system("mkdir " + cryoCAREpath)
		if not os.path.exists(os.path.join(cryoCAREpath, "rawTomos")) or not os.path.isdir(os.path.join(cryoCAREpath, "rawTomos")):
			os.system("mkdir " + os.path.join(cryoCAREpath, "rawTomos"))
		os.system("mv " + os.path.join(oddPath, "odd_" + alignedName) + " " + os.path.join(cryoCAREpath, "rawTomos"))
		os.system("mv " + os.path.join(evenPath, "even_" + alignedName) + " " + os.path.join(cryoCAREpath, "rawTomos"))
  
		#Establish some order
		os.chdir(startingDirectory)
		os.system("mkdir newTomo")
		os.system("mv " + ctf_stackOutFileNameFull + " newTomo")
		os.system("mv " + alnFileName + " newTomo")
		os.system("mv " + tomoName + "* " + "newTomo")
		if include_CTFcorrection:
			os.system("mv " + outDefFile + " newTomo")
		os.system("mv newTomo rec_" + tomoName)



def main():
	if reconstructHalfMaps:
		rec(apix, expectedDefocus, TiltAxisRotation)
	
	tomosToUseTrain = []
	tomosToUsePredict = []

	mdocFiles_train = glob.glob(os.path.join(startingDirectory, mdocryoCARETrain))
	mdocFiles_predict = glob.glob(os.path.join(startingDirectory, mdocryoCAREPredict))

	for files in mdocFiles_train:
		tomoNamex = "out_" + files.split("/")[-1].split(".")[0]
		tomosToUseTrain.append(tomoNamex)	

	for files in mdocFiles_predict:
		tomoNamex = "out_" + files.split("/")[-1].split(".")[0]
		tomosToUsePredict.append(tomoNamex)	
  
	if trainCryoCARE: 
		if len(tomosToUseTrain) == 0:
			print("NO MDOCS FOUND FOR CRYOCARE TRAINING. EXIT.")
			return     
		trainForAllTomos.train(cryoCAREpath, tomosToUseTrain)
   
	if predictCryoCARE:
		if len(tomosToUsePredict) == 0:
			print("NO MDOCS FOUND FOR CRYOCARE PREDICTION. EXIT.")
			return     
		predictForAllTomos.predict(cryoCAREpath, tomosToUsePredict)


main()
