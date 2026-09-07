{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
{
  packages = [
    pkgs.git
    pkgs.cmake
    pkgs.wayland
    pkgs.wayland-scanner
    pkgs.libxkbcommon
    pkgs.libffi
    pkgs.glfw
    pkgs.waylandpp
    pkgs.gcc
  ];
  languages.cplusplus.enable = true;
  env.LD_LIBRARY_PATH = "${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib";
}
