# CMake generated Testfile for 
# Source directory: /home/iosmichael/Documents/slam/glog
# Build directory: /home/iosmichael/Documents/slam/glog/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(demangle "/home/iosmichael/Documents/slam/glog/build/demangle_unittest")
add_test(logging "/home/iosmichael/Documents/slam/glog/build/logging_unittest")
add_test(signalhandler "/home/iosmichael/Documents/slam/glog/build/signalhandler_unittest")
add_test(stacktrace "/home/iosmichael/Documents/slam/glog/build/stacktrace_unittest")
set_tests_properties(stacktrace PROPERTIES  TIMEOUT "30")
add_test(stl_logging "/home/iosmichael/Documents/slam/glog/build/stl_logging_unittest")
add_test(symbolize "/home/iosmichael/Documents/slam/glog/build/symbolize_unittest")
