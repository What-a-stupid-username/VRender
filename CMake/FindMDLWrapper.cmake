#
# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Locate the MDL-Wrapper inside the support folder

find_library(mdl_wrapper_LIBRARY
  NAMES mdl_wrapper
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/support/mdl_wrapper/lib
  NO_DEFAULT_PATH
  )

find_path(mdl_wrapper_INCLUDE_DIR
  NAMES mdl_wrapper.h
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/support/mdl_wrapper/include
  NO_DEFAULT_PATH
  )

if( mdl_wrapper_LIBRARY AND
    mdl_wrapper_INCLUDE_DIR
    )
  set(mdl_wrapper_FOUND TRUE)

  add_library(mdl_wrapper UNKNOWN IMPORTED)
  set_property(TARGET mdl_wrapper PROPERTY IMPORTED_LOCATION "${mdl_wrapper_LIBRARY}")
endif()
