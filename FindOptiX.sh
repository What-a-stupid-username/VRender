# !/bin/sh
if  [ ! -n "$1" ] ;then
	read -p "Input your OptiX install Dir:" -r Path
else
    Path=$1
fi
Input_Path=$Path
PAN=${Path:0:1}
typeset -l pan
pan=$PAN
Path=${Path:1:${#Path}-1}
Path=${Path//\\//}
Path=${Path//:/}
Path="/$pan$Path"
CMAKE_PATH="$Path/SDK/CMake"
if [ -d "$CMAKE_PATH" ]; then
	cp -r "$CMAKE_PATH" ./CMake
else
	err="OptiX not fond in: \n $CMAKE_PATH"
	echo -e "\033[41;37m ${err} \033[0m"
	exit
fi
echo Done!

FindOptiX_cmake="./CMake/FindOptiX.cmake"
form_str="s#set(OptiX_INSTALL_DIR \"\${CMAKE_SOURCE_DIR}/../\" CACHE PATH \"Path to OptiX installed location.\")#"

new_str=${Input_Path//\\//}
new_str="set(OptiX_INSTALL_DIR \"$new_str\")"
form_str="$form_str$new_str#g"
sed -i "$form_str" "$FindOptiX_cmake"


# "C:\ProgramData\NVIDIA Corporation\OptiX SDK 5.1.0"
# "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.1.0"
