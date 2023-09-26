# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/davidadeshina/Documents/Git-projects/SMLF-Library

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build

# Include any dependencies generated for this target.
include OurDataframe/CMakeFiles/ODf.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include OurDataframe/CMakeFiles/ODf.dir/compiler_depend.make

# Include the progress variables for this target.
include OurDataframe/CMakeFiles/ODf.dir/progress.make

# Include the compile flags for this target's objects.
include OurDataframe/CMakeFiles/ODf.dir/flags.make

OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.o: OurDataframe/CMakeFiles/ODf.dir/flags.make
OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.o: /Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe/ODf.cpp
OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.o: OurDataframe/CMakeFiles/ODf.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.o"
	cd /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/OurDataframe && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.o -MF CMakeFiles/ODf.dir/ODf.cpp.o.d -o CMakeFiles/ODf.dir/ODf.cpp.o -c /Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe/ODf.cpp

OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ODf.dir/ODf.cpp.i"
	cd /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/OurDataframe && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe/ODf.cpp > CMakeFiles/ODf.dir/ODf.cpp.i

OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ODf.dir/ODf.cpp.s"
	cd /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/OurDataframe && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe/ODf.cpp -o CMakeFiles/ODf.dir/ODf.cpp.s

# Object files for target ODf
ODf_OBJECTS = \
"CMakeFiles/ODf.dir/ODf.cpp.o"

# External object files for target ODf
ODf_EXTERNAL_OBJECTS =

OurDataframe/libODf.dylib: OurDataframe/CMakeFiles/ODf.dir/ODf.cpp.o
OurDataframe/libODf.dylib: OurDataframe/CMakeFiles/ODf.dir/build.make
OurDataframe/libODf.dylib: OurDataframe/CMakeFiles/ODf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libODf.dylib"
	cd /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/OurDataframe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ODf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
OurDataframe/CMakeFiles/ODf.dir/build: OurDataframe/libODf.dylib
.PHONY : OurDataframe/CMakeFiles/ODf.dir/build

OurDataframe/CMakeFiles/ODf.dir/clean:
	cd /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/OurDataframe && $(CMAKE_COMMAND) -P CMakeFiles/ODf.dir/cmake_clean.cmake
.PHONY : OurDataframe/CMakeFiles/ODf.dir/clean

OurDataframe/CMakeFiles/ODf.dir/depend:
	cd /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/davidadeshina/Documents/Git-projects/SMLF-Library /Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/OurDataframe /Users/davidadeshina/Documents/Git-projects/SMLF-Library/build/OurDataframe/CMakeFiles/ODf.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : OurDataframe/CMakeFiles/ODf.dir/depend

