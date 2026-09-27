cmake_minimum_required(VERSION 3.0.2)
project(Full_door_v4)
find_package(catkin REQUIRED)
catkin_package()
install(DIRECTORY urdf meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
