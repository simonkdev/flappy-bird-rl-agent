
if (NOT EXISTS "/home/simonkdev/Documents/02-IT/flappy-bird-rl-agent/flappy-bird/build_check/caffeine-gl/external/glfw/install_manifest.txt")
    message(FATAL_ERROR "Cannot find install manifest: \"/home/simonkdev/Documents/02-IT/flappy-bird-rl-agent/flappy-bird/build_check/caffeine-gl/external/glfw/install_manifest.txt\"")
endif()

file(READ "/home/simonkdev/Documents/02-IT/flappy-bird-rl-agent/flappy-bird/build_check/caffeine-gl/external/glfw/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")

foreach (file ${files})
  message(STATUS "Uninstalling \"$ENV{DESTDIR}${file}\"")
  if (EXISTS "$ENV{DESTDIR}${file}")
    exec_program("/nix/store/k1b82rs72af124wmq6nb00nhw5i7ikp7-cmake-4.1.2/bin/cmake" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
                 OUTPUT_VARIABLE rm_out
                 RETURN_VALUE rm_retval)
    if (NOT "${rm_retval}" STREQUAL 0)
      MESSAGE(FATAL_ERROR "Problem when removing \"$ENV{DESTDIR}${file}\"")
    endif()
  elseif (IS_SYMLINK "$ENV{DESTDIR}${file}")
    EXEC_PROGRAM("/nix/store/k1b82rs72af124wmq6nb00nhw5i7ikp7-cmake-4.1.2/bin/cmake" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
                 OUTPUT_VARIABLE rm_out
                 RETURN_VALUE rm_retval)
    if (NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing symlink \"$ENV{DESTDIR}${file}\"")
    endif()
  else()
    message(STATUS "File \"$ENV{DESTDIR}${file}\" does not exist.")
  endif()
endforeach()

