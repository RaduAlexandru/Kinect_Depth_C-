# Computer Vision Lecture MA-INF 2201
#
# CMakeLists.txt for exercise 00
# see http://www.cmake.org/ for a documentation about CMake
#
# compile your programm using the following commands:
#
# !!! To Make an ordinary Unix Makefile project you can use the Generator in the following way
# cmake -G"Unix Makefiles"
# make

# !!! NOTE
# With the way described above you will be sure that 
# created cmake files  will be stored in build folder ,and 
# binary files will be stored in the bin folder
# which you should not version while using the SVN

cmake_minimum_required(VERSION 2.6)
project (exercise00)

SET(CMAKE_CXX_FLAGS "-std=c++0x")				    # New C11
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)		# Binary file location
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})		# Exectuable file location for main and subprojects 
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)	# Library file location for main and subprojects
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR})			# Source folder
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR})		# Header folder

#set(CMAKE_BUILD_TYPE debug)					# choose if you want to compile with or without debugging flags,
set(CMAKE_BUILD_TYPE release)					# uncomment only one of the following two commands at a time

#To build the project openCV is required
find_package( OpenCV REQUIRED )

#Include Required Directories
include_directories(${PROJECT_INCLUDE_DIR})			
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


# Create a shared library (To use library creation uncomment the following)
#	add_library(	
#		LIBRARY_NAME 
#		${PROJECT_SOURCE_DIR}/some.cpp
#		${PROJECT_SOURCE_DIR}/someMore.cpp
#		)

#Create an Executable for segment
add_executable(sheet00 ${PROJECT_SOURCE_DIR}/sheet00.cpp)	#Add executable to sheet00.cpp
target_link_libraries( sheet00 ${OpenCV_LIBS} )			#Link the executable with opencv library
 
