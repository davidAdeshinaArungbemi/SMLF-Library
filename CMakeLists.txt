cmake_minimum_required(VERSION 3.25)
project(SMLF)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(OurDataframe)
add_subdirectory(examples)

add_library(SMLF INTERFACE)

# target_sources(SMLF INTERFACE
#     LinearRegression.hpp
# )

target_include_directories(SMLF INTERFACE 
                ${CMAKE_CURRENT_SOURCE_DIR}/include  
                ${CMAKE_CURRENT_SOURCE_DIR}/OurDataframe
)

target_link_libraries(SMLF INTERFACE ODf)


# add_library()
